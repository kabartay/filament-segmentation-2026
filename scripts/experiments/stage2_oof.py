"""Out-of-fold evaluation of stage 2 (``filseg.stage2``) on the submitted tree pipeline.

Steps (each reads what the previous one wrote under runs/stage2/):

  crops  the stage-1 selection of every training image, out of fold (the hurdle scorer applied to a fold is fitted
         on the other four, as in bootstrap_pq.py), its seeds recomputed from the tree levels, and the crop inputs
         and soft targets of every seed -> crops.npy, targets.npy, meta.pkl
  train  one refiner per fold, trained on the other folds' crops (only seeds some annotator drew), predicting its
         own fold's crops -> <tag>_f{k}.pt, <tag>_pred.npy (all folds filled in)
  eval   paste-back at several pixel thresholds, pooled PQ against every annotator entry, paired bootstrap against
         stage 1 alone (the submitted pipeline, OOF 0.4267) -> <tag>_eval.json
  jitter extra training crops: tree nodes at other levels than the selected seed (fragments, thinner or wider
         versions of a filament), matched to an annotator mask by IoU > DREW_IOU or by containment >= CONTAIN,
         near-duplicates (IoU > JIT_DEDUP with a seed or a kept node) skipped -> jit_crops.npy, jit_targets.npy,
         jit_meta.pkl. Only for training (``train --extra jit``); evaluation stays on the selected seeds.
  blend  average the OOF crop predictions of several tags -> <out>_pred.npy, to evaluate an ensemble of refiners
  export the fold refiners as float16 state dicts for inference -> weights/stage2_<tag>_f{k}.pt (about 49 MB each;
         at threshold 0.40 about 1 crop pixel in 10^5 changes against float32)

Run (FILSEG_CACHE=data/cache1536, PYTHONPATH=.):
  .venv/bin/python scripts/experiments/stage2_oof.py crops --workers 4
  .venv/bin/python scripts/experiments/stage2_oof.py train --tag unet --arch Unet --epochs 30
  .venv/bin/python scripts/experiments/stage2_oof.py eval --tag unet --workers 4
  .venv/bin/python scripts/experiments/stage2_oof.py export --tag unet
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # shared machine: see bootstrap_pq.py

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util

from filseg.cache import CACHE
from filseg.data import ROOT, TRAIN_IMAGES, group_by_stem, load_train, read_image
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import components, upsample
from filseg.scorer import CANDIDATE_CLOSE_PX, CANDIDATE_MIN_AREA
from filseg.stage2 import (CHANNELS, CROP_H, CROP_W, MIN_SCALE, Frame, build_refiner, crop_inputs, crop_one,
                           crop_target, paste_back, predict_crops, to_tensor, to_u8, unwarp)
from filseg.train import RUNS, soft_dice_loss
from filseg.tree import add_context, antichain, fit_hurdle
from filseg.tree import training_table as tree_table

NODES = RUNS / "oof_tree12s_1536_v4.pkl"
PROBS_RUN = "unet_r34_1536"
TAU = 0.26
OUT = RUNS / "stage2"
CONTAIN = 0.5  # jitter: a node with at least this share of its area inside an annotator's filament is a fragment of it
JIT_DEDUP = 0.85  # jitter: nodes this similar (IoU) to a selected seed or to an already-kept node are skipped
JIT_PER_SEED = 4  # jitter: at most this many extra crops per selected seed of the image (nodes visited in random order)
G: dict = {}


def oof_selection() -> list[dict]:
    """Per image: stem, fold, levels, node counts per level, chosen node indices and their scores.

    Cached in runs/stage2/selection.pkl (the CV hurdle fits take minutes); delete it to recompute.
    """
    cache = OUT / "selection.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    data = pickle.load(open(NODES, "rb"))
    for d in data:
        add_context(d["feats"], d["parent"])
    _, X, y, fold, w = tree_table(data)
    score = np.zeros_like(y)
    for f in np.unique(fold):
        h, m = fit_hurdle(X[fold != f], y[fold != f], w[fold != f])
        score[fold == f] = h.predict_proba(X[fold == f])[:, 1] * m.predict(X[fold == f])
    out, start = [], 0
    for d in data:
        n = len(d["feats"])
        s = score[start:start + n]
        start += n
        chosen = sorted(antichain(s, d["parent"], TAU))
        out.append(dict(stem=d["stem"], fold=d["fold"], level=np.asarray(d["level"]), chosen=chosen,
                        score=s[chosen]))
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(cache, "wb"))
    return out


def _init_crops() -> None:
    G["index"] = json.loads((CACHE / "index.json").read_text())["train"]
    G["images"] = np.load(CACHE / "train_images.npy", mmap_mode="r")
    G["groups"] = group_by_stem(load_train())
    G["probs"] = {f: np.load(RUNS / f"{PROBS_RUN}_f{f}" / "val_probs_v4.npy", mmap_mode="r") for f in range(5)}
    G["pos"] = {f: {s: i for i, s in enumerate(json.loads((RUNS / f"{PROBS_RUN}_f{f}" / "val_stems.json").read_text()))}
                for f in range(5)}


def _crops_one(sel: dict) -> dict:
    stem, f = sel["stem"], sel["fold"]
    full = upsample(np.asarray(G["probs"][f][G["pos"][f][stem]], np.float32))
    disk = fit_disk(np.array(G["images"][G["index"].index(stem)]))
    levels = sorted(set(float(v) for v in sel["level"]))
    rles = []
    for lv in levels:  # the node numbering of filseg.tree.tree_candidates
        rles += [c.rle for c in components(full, lv, CANDIDATE_CLOSE_PX, disk, min_area=CANDIDATE_MIN_AREA)]
    assert len(rles) == len(sel["level"]), f"{stem}: {len(rles)} nodes recomputed, {len(sel['level'])} cached"
    seeds = [mask_util.decode(rles[j]).astype(bool) for j in sel["chosen"]]
    image = read_image(TRAIN_IMAGES / f"{stem}.jpeg")
    x, frames = crop_inputs(image, full, seeds)
    entries = [e.rles for e in G["groups"][stem]]
    tgt, drew = [], []
    for s, fr in zip(seeds, frames):
        t, k = crop_target(s, entries, fr)
        tgt.append(t)
        drew.append(k)
    return dict(stem=stem, fold=f, x=x, y=np.array(tgt, np.uint8).reshape(len(seeds), CROP_H, CROP_W),
                M=[fr.M for fr in frames], scale=[fr.scale for fr in frames], drew=drew,
                seeds=[rles[j] for j in sel["chosen"]], score=sel["score"])


def step_crops(workers: int) -> None:
    t0 = time.time()
    sels = oof_selection()
    print(f"selection: {len(sels)} images, {sum(len(s['chosen']) for s in sels)} seeds, {time.time() - t0:.0f} s",
          flush=True)
    with ProcessPoolExecutor(workers, initializer=_init_crops) as ex:
        recs = list(ex.map(_crops_one, sels, chunksize=2))
    n = sum(len(r["drew"]) for r in recs)
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(OUT / "crops.npy", "w+", np.uint8, (n, len(CHANNELS), CROP_H, CROP_W))
    Y = np.lib.format.open_memmap(OUT / "targets.npy", "w+", np.uint8, (n, CROP_H, CROP_W))
    meta, row = [], 0
    for r in recs:
        k = len(r["drew"])
        if k:
            X[row:row + k], Y[row:row + k] = r["x"], r["y"]
        meta.append(dict(stem=r["stem"], fold=r["fold"], row=row, n=k, M=r["M"], scale=r["scale"], drew=r["drew"],
                         seeds=r["seeds"], score=r["score"]))
        row += k
    X.flush()
    Y.flush()
    pickle.dump(meta, open(OUT / "meta.pkl", "wb"))
    drew = np.concatenate([m["drew"] for m in meta if m["n"]])
    print(f"{n} crops, {int((drew > 0).sum())} drawn by some annotator, {time.time() - t0:.0f} s -> {OUT}", flush=True)


def _jitter_one(sel: dict) -> dict:
    stem, f = sel["stem"], sel["fold"]
    full = upsample(np.asarray(G["probs"][f][G["pos"][f][stem]], np.float32))
    disk = fit_disk(np.array(G["images"][G["index"].index(stem)]))
    levels = sorted(set(float(v) for v in sel["level"]))
    rles = []
    for lv in levels:  # the node numbering of filseg.tree.tree_candidates (lowest level first)
        rles += [c.rle for c in components(full, lv, CANDIDATE_CLOSE_PX, disk, min_area=CANDIDATE_MIN_AREA)]
    assert len(rles) == len(sel["level"]), f"{stem}: {len(rles)} nodes recomputed, {len(sel['level'])} cached"
    chosen = set(sel["chosen"])
    seeds = [mask_util.decode(rles[j]).astype(bool) for j in sel["chosen"]]
    entries = [e.rles for e in G["groups"][stem]]
    gt = [r for rl in entries for r in rl]
    rest = [j for j in range(len(rles)) if j not in chosen]
    out = dict(stem=stem, fold=f, x=[], y=[], drew=[], level=[])
    if not rest or not gt:
        return out
    near_seed = (np.asarray(mask_util.iou([rles[j] for j in rest], [rles[j] for j in sel["chosen"]],
                                          [0] * len(chosen))).reshape(len(rest), -1).max(axis=1)
                 if chosen else np.zeros(len(rest)))
    kept: list[dict] = []
    cap = JIT_PER_SEED * max(1, len(seeds))
    order = np.random.default_rng(int(stem[:14])).permutation(len(rest))
    rest, near_seed = [rest[i] for i in order], near_seed[order]
    image = read_image(TRAIN_IMAGES / f"{stem}.jpeg")
    prob8 = to_u8(full)
    for j, ns in zip(rest, near_seed):
        if len(kept) >= cap:
            break
        if ns > JIT_DEDUP:
            continue
        if kept and np.asarray(mask_util.iou([rles[j]], kept, [0] * len(kept))).max() > JIT_DEDUP:
            continue
        node = mask_util.decode(rles[j]).astype(bool)
        others = np.zeros_like(node)
        for sd in seeds:
            if not (sd & node).any():  # seeds overlapping the node are versions of the same filament
                others |= sd
        x, fr = crop_one(image, prob8, node, others)
        t, k = crop_target(node, entries, fr, contain=CONTAIN)
        if k == 0:
            continue
        kept.append(rles[j])
        out["x"].append(x)
        out["y"].append(t)
        out["drew"].append(k)
        out["level"].append(float(sel["level"][j]))
    return out


def step_jitter(workers: int) -> None:
    t0 = time.time()
    sels = oof_selection()
    with ProcessPoolExecutor(workers, initializer=_init_crops) as ex:
        recs = list(ex.map(_jitter_one, sels, chunksize=2))
    n = sum(len(r["drew"]) for r in recs)
    X = np.lib.format.open_memmap(OUT / "jit_crops.npy", "w+", np.uint8, (n, len(CHANNELS), CROP_H, CROP_W))
    Y = np.lib.format.open_memmap(OUT / "jit_targets.npy", "w+", np.uint8, (n, CROP_H, CROP_W))
    meta, row = [], 0
    for r in recs:
        k = len(r["drew"])
        if k:
            X[row:row + k], Y[row:row + k] = np.stack(r["x"]), np.stack(r["y"])
        meta.append(dict(stem=r["stem"], fold=r["fold"], row=row, n=k, drew=r["drew"], level=r["level"]))
        row += k
    X.flush()
    Y.flush()
    pickle.dump(meta, open(OUT / "jit_meta.pkl", "wb"))
    print(f"{n} extra crops from {sum(1 for m in meta if m['n'])} images, {time.time() - t0:.0f} s -> {OUT}", flush=True)


def rows_of(meta: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per crop: fold, number of annotators who drew it, scale."""
    fold = np.concatenate([[m["fold"]] * m["n"] for m in meta if m["n"]])
    drew = np.concatenate([m["drew"] for m in meta if m["n"]])
    scale = np.concatenate([m["scale"] for m in meta if m["n"]])
    return fold, drew, scale


def augment(x: torch.Tensor, y: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample flips and a small affine jitter (rotation +-10 deg, scale 0.9-1.1, shift +-10 px), applied
    identically to the input channels and the target with one bilinear resampling."""
    n = x.shape[0]
    ang = np.deg2rad(rng.uniform(-10, 10, n))
    sc = rng.uniform(0.9, 1.1, n)
    fx = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    fy = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    tx, ty = rng.uniform(-10, 10, n), rng.uniform(-10, 10, n)
    hw, hh = CROP_W / 2, CROP_H / 2
    theta = np.zeros((n, 2, 3), np.float32)  # output -> input sampling, in normalised coordinates
    cos, sin = np.cos(ang) / sc, np.sin(ang) / sc
    theta[:, 0, 0] = cos * fx
    theta[:, 0, 1] = -sin * fy * hh / hw
    theta[:, 1, 0] = sin * fx * hw / hh
    theta[:, 1, 1] = cos * fy
    theta[:, 0, 2] = tx / hw
    theta[:, 1, 2] = ty / hh
    th = torch.from_numpy(theta).to(x.device)
    both = torch.cat([x, y], dim=1)
    grid = F.affine_grid(th, list(both.shape), align_corners=False)
    both = F.grid_sample(both, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return both[:, :-1], both[:, -1:]


def step_train(tag: str, arch: str, encoder: str, epochs: int, batch: int, lr: float, folds: list[int],
               extra: str | None = None, aug: bool = False) -> None:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    X = np.load(OUT / "crops.npy", mmap_mode="r")
    Y = np.load(OUT / "targets.npy", mmap_mode="r")
    meta = pickle.load(open(OUT / "meta.pkl", "rb"))
    fold, drew, _ = rows_of(meta)
    if extra:  # extra training crops (step jitter); never predicted or evaluated
        XE = np.load(OUT / f"{extra}_crops.npy", mmap_mode="r")
        YE = np.load(OUT / f"{extra}_targets.npy", mmap_mode="r")
        fold_e = np.concatenate([[m["fold"]] * m["n"] for m in pickle.load(open(OUT / f"{extra}_meta.pkl", "rb"))
                                 if m["n"]])
    pred_path = OUT / f"{tag}_pred.npy"
    pred = (np.load(pred_path) if pred_path.exists() else np.zeros((len(X), CROP_H, CROP_W), np.float16))
    log = open(OUT / f"{tag}_train.log", "a")
    for f in folds:
        torch.manual_seed(f)
        rng = np.random.default_rng(f)
        tr = np.nonzero((fold != f) & (drew > 0))[0]
        Xtr, Ytr = np.asarray(X[tr]), np.asarray(Y[tr])
        if extra:
            te_ = np.nonzero(fold_e != f)[0]
            Xtr, Ytr = np.concatenate([Xtr, np.asarray(XE[te_])]), np.concatenate([Ytr, np.asarray(YE[te_])])
            tr = np.arange(len(Xtr))
        model = build_refiner(arch, encoder).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        steps = epochs * (len(tr) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            perm = rng.permutation(len(tr))
            run = 0.0
            for b in range(len(tr) // batch):
                idx = perm[b * batch:(b + 1) * batch]
                x = to_tensor(Xtr[idx])
                y = torch.from_numpy(Ytr[idx]).float()[:, None] / 255.0
                if aug:
                    x, y = augment(x.to(device), y.to(device), rng)
                else:
                    if rng.random() < 0.5:
                        x, y = x.flip(-1), y.flip(-1)
                    if rng.random() < 0.5:
                        x, y = x.flip(-2), y.flip(-2)
                    x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y) + 0.5 * soft_dice_loss(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                run += loss.item()
            line = f"{time.strftime('%H:%M:%S')} fold {f} epoch {ep + 1} loss {run / max(1, len(tr) // batch):.4f} {time.time() - t0:.0f}s"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
        torch.save(model.state_dict(), OUT / f"{tag}_f{f}.pt")
        te = np.nonzero(fold == f)[0]
        pred[te] = predict_crops([model], np.asarray(X[te]), device).astype(np.float16)
        np.save(pred_path, pred)
        print(f"fold {f}: {len(tr)} training crops, {len(te)} predicted, {time.time() - t0:.0f} s", flush=True)


def _init_eval(tag: str) -> None:
    G["groups"] = group_by_stem(load_train())
    G["pred"] = np.load(OUT / f"{tag}_pred.npy", mmap_mode="r")


def _eval_one(item: tuple[dict, tuple[float, ...]]) -> tuple[str, dict]:
    m, thresholds = item
    entries = G["groups"][m["stem"]]
    seeds = [mask_util.decode(r).astype(bool) for r in m["seeds"]]
    out = {}

    def counts(masks: list[np.ndarray]) -> list[float]:
        rles = [mask_util.encode(np.asfortranarray(x.astype(np.uint8))) for x in masks]
        c = PQCounts()
        for e in entries:
            c.add(iou_matrix(e.rles, rles), len(e.rles), len(rles))
        return [c.tp_iou_sum, c.tp, c.fp, c.fn]

    out["seed"] = counts(seeds)
    refined = []
    for i in range(m["n"]):
        if m["scale"][i] < MIN_SCALE:
            refined.append(None)
        else:
            refined.append(unwarp(np.asarray(G["pred"][m["row"] + i], np.float32), Frame(m["M"][i], m["scale"][i])))
    for t in thresholds:
        out[f"t{t:.2f}"] = counts(paste_back(seeds, refined, np.asarray(m["score"]), t))
    return m["stem"], out


def pq(m: np.ndarray) -> np.ndarray:
    s = m.sum(axis=-2)
    return s[..., 0] / (s[..., 1] + 0.5 * s[..., 2] + 0.5 * s[..., 3])


def step_eval(tag: str, workers: int, thresholds: tuple[float, ...], name: str = "eval") -> None:
    meta = pickle.load(open(OUT / "meta.pkl", "rb"))
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init_eval, initargs=(tag,)) as ex:
        res = dict(ex.map(_eval_one, [(m, thresholds) for m in meta], chunksize=2))
    stems = sorted(res)
    fold = np.array([next(m["fold"] for m in meta if m["stem"] == s) for s in stems])
    keys = list(res[stems[0]])
    A = {k: np.array([res[s][k] for s in stems]) for k in keys}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(stems), size=(2000, len(stems)))
    base = pq(A["seed"][idx])
    groups = group_by_stem(load_train())
    multi = np.array([len(groups[s]) >= 2 for s in stems])

    def sq_rq(m: np.ndarray) -> tuple[float, float]:
        t = m.sum(axis=0)
        return float(t[0] / t[1]), float(t[1] / (t[1] + 0.5 * t[2] + 0.5 * t[3]))

    report = {"images": len(stems), "multi_annotator_images": int(multi.sum()), "seconds": round(time.time() - t0)}
    for k in keys:
        b = pq(A[k][idx])
        per_fold = [float(pq(A[k][fold == f])) for f in range(5)]
        sq, rq = sq_rq(A[k])
        msq, mrq = sq_rq(A[k][multi])
        report[k] = {"pq": round(float(pq(A[k])), 5), "sq": round(sq, 4), "rq": round(rq, 4),
                     "pq_ci95": [round(float(v), 4) for v in np.percentile(b, [2.5, 97.5])],
                     "per_fold": [round(v, 4) for v in per_fold],
                     "per_fold_mean_sd": [round(float(np.mean(per_fold)), 4), round(float(np.std(per_fold, ddof=1)), 4)],
                     "multi": {"pq": round(float(pq(A[k][multi])), 4), "sq": round(msq, 4), "rq": round(mrq, 4)},
                     "gain": round(float(pq(A[k]) - pq(A["seed"])), 5),
                     "gain_ci95": [round(float(v), 5) for v in np.percentile(b - base, [2.5, 97.5])],
                     "gain_folds_positive": int(sum(pq(A[k][fold == f]) > pq(A["seed"][fold == f]) for f in range(5)))}
    (OUT / f"{tag}_{name}.json").write_text(json.dumps(report, indent=1))
    np.savez(OUT / f"{tag}_{name}_per_image.npz", stems=np.array(stems), fold=fold, **A)
    print(json.dumps(report, indent=1))


def step_compare(new: str, old: str, n_boot: int = 2000) -> None:
    """Paired bootstrap of two evaluated variants over the same images.

    Args:
        new: ``<tag>_<name>:<key>``, e.g. ``ens_pi:t0.40`` (per-image file from step eval).
        old: the baseline in the same form, e.g. ``unet_pi:t0.40``.
        n_boot: bootstrap resamples of images.
    """
    def load(spec: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f, k = spec.split(":")
        z = np.load(OUT / f"{f}_per_image.npz")
        return z["stems"], z["fold"], z[k]

    sa, fold, a = load(new)
    sb, _, b = load(old)
    assert (sa == sb).all(), "per-image files cover different images"
    idx = np.random.default_rng(0).integers(0, len(sa), size=(n_boot, len(sa)))
    d = pq(a[idx]) - pq(b[idx])
    per_fold = [round(float(pq(a[fold == f]) - pq(b[fold == f])), 4) for f in range(5)]
    report = {"new": new, "old": old, "pq_new": round(float(pq(a)), 5), "pq_old": round(float(pq(b)), 5),
              "gain": round(float(pq(a) - pq(b)), 5),
              "gain_ci95": [round(float(v), 5) for v in np.percentile(d, [2.5, 97.5])],
              "gain_per_fold": per_fold, "folds_positive": int(sum(v > 0 for v in per_fold))}
    print(json.dumps(report), flush=True)


def step_blend(tags: list[str], out: str) -> None:
    preds = [np.load(OUT / f"{t}_pred.npy", mmap_mode="r") for t in tags]
    acc = np.zeros(preds[0].shape, np.float32)
    for lo in range(0, len(acc), 256):
        for q in preds:
            acc[lo:lo + 256] += np.asarray(q[lo:lo + 256], np.float32) / len(preds)
    np.save(OUT / f"{out}_pred.npy", acc.astype(np.float16))
    print(f"{' + '.join(tags)} -> {OUT / f'{out}_pred.npy'}", flush=True)


def step_export(tag: str, folds: list[int]) -> None:
    for f in folds:
        state = torch.load(OUT / f"{tag}_f{f}.pt", map_location="cpu")
        out = ROOT / "weights" / f"stage2_{tag}_f{f}.pt"
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in state.items()}, out)
        print(f"fold {f} -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["crops", "jitter", "train", "eval", "blend", "export", "compare"])
    ap.add_argument("--extra", default=None, help="train: extra crop set to add, e.g. jit (from step jitter)")
    ap.add_argument("--aug", action="store_true", help="train: per-sample flips and small affine jitter")
    ap.add_argument("--blend", default="", help="blend: comma-separated tags to average into --tag")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="unet")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55")
    ap.add_argument("--name", default="eval", help="eval: report written to <tag>_<name>.json")
    ap.add_argument("--new", default="", help="compare: <tag>_<name>:<key> of the variant, e.g. ens_pi:t0.40")
    ap.add_argument("--old", default="", help="compare: the baseline in the same form, e.g. unet_pi:t0.40")
    a = ap.parse_args()
    if a.step == "crops":
        step_crops(a.workers)
    elif a.step == "train":
        step_train(a.tag, a.arch, a.encoder, a.epochs, a.batch, a.lr, [int(k) for k in a.folds.split(",")],
                   a.extra, a.aug)
    elif a.step == "jitter":
        step_jitter(a.workers)
    elif a.step == "compare":
        step_compare(a.new, a.old)
    elif a.step == "blend":
        step_blend(a.blend.split(","), a.tag)
    elif a.step == "export":
        step_export(a.tag, [int(k) for k in a.folds.split(",")])
    else:
        step_eval(a.tag, a.workers, tuple(float(t) for t in a.thresholds.split(",")), a.name)


if __name__ == "__main__":
    main()
