"""Out-of-fold evaluation of stage 2 (``filseg.stage2``) on the submitted tree pipeline.

Steps (each reads what the previous one wrote under runs/stage2/):

  crops  the stage-1 selection of every training image, out of fold (the hurdle scorer applied to a fold is fitted
         on the other four, as in bootstrap_pq.py), its seeds recomputed from the tree levels, and the crop inputs
         and soft targets of every seed -> crops.npy, targets.npy, meta.pkl
  train  one refiner per fold, trained on the other folds' crops (only seeds some annotator drew), predicting its
         own fold's crops -> <tag>_f{k}.pt, <tag>_pred.npy (all folds filled in)
  eval   paste-back at several pixel thresholds, pooled PQ against every annotator entry, paired bootstrap against
         stage 1 alone (the submitted pipeline, OOF 0.4267) -> <tag>_eval.json

Run (FILSEG_CACHE=data/cache1536, PYTHONPATH=.):
  .venv/bin/python scripts/experiments/stage2_oof.py crops --workers 4
  .venv/bin/python scripts/experiments/stage2_oof.py train --tag unet --arch Unet --epochs 30
  .venv/bin/python scripts/experiments/stage2_oof.py eval --tag unet --workers 4
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
from filseg.data import TRAIN_IMAGES, group_by_stem, load_train, read_image
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import components, upsample
from filseg.scorer import CANDIDATE_CLOSE_PX, CANDIDATE_MIN_AREA
from filseg.stage2 import (CHANNELS, CROP_H, CROP_W, MIN_SCALE, Frame, build_refiner, crop_inputs, crop_target,
                           paste_back, predict_crops, to_tensor, unwarp)
from filseg.train import RUNS, soft_dice_loss
from filseg.tree import add_context, antichain, fit_hurdle
from filseg.tree import training_table as tree_table

NODES = RUNS / "oof_tree12s_1536_v4.pkl"
PROBS_RUN = "unet_r34_1536"
TAU = 0.26
OUT = RUNS / "stage2"
G: dict = {}


def oof_selection() -> list[dict]:
    """Per image: stem, fold, levels, node counts per level, chosen node indices and their scores."""
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


def rows_of(meta: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per crop: fold, number of annotators who drew it, scale."""
    fold = np.concatenate([[m["fold"]] * m["n"] for m in meta if m["n"]])
    drew = np.concatenate([m["drew"] for m in meta if m["n"]])
    scale = np.concatenate([m["scale"] for m in meta if m["n"]])
    return fold, drew, scale


def step_train(tag: str, arch: str, encoder: str, epochs: int, batch: int, lr: float, folds: list[int]) -> None:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    X = np.load(OUT / "crops.npy", mmap_mode="r")
    Y = np.load(OUT / "targets.npy", mmap_mode="r")
    meta = pickle.load(open(OUT / "meta.pkl", "rb"))
    fold, drew, _ = rows_of(meta)
    pred_path = OUT / f"{tag}_pred.npy"
    pred = (np.load(pred_path) if pred_path.exists() else np.zeros((len(X), CROP_H, CROP_W), np.float16))
    log = open(OUT / f"{tag}_train.log", "a")
    for f in folds:
        torch.manual_seed(f)
        rng = np.random.default_rng(f)
        tr = np.nonzero((fold != f) & (drew > 0))[0]
        Xtr, Ytr = np.asarray(X[tr]), np.asarray(Y[tr])
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


def step_eval(tag: str, workers: int, thresholds: tuple[float, ...]) -> None:
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
    report = {"images": len(stems), "seconds": round(time.time() - t0)}
    for k in keys:
        b = pq(A[k][idx])
        report[k] = {"pq": round(float(pq(A[k])), 5),
                     "per_fold": [round(float(pq(A[k][fold == f])), 4) for f in range(5)],
                     "gain": round(float(pq(A[k]) - pq(A["seed"])), 5),
                     "gain_ci95": [round(float(v), 5) for v in np.percentile(b - base, [2.5, 97.5])]}
    (OUT / f"{tag}_eval.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["crops", "train", "eval"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="unet")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55")
    a = ap.parse_args()
    if a.step == "crops":
        step_crops(a.workers)
    elif a.step == "train":
        step_train(a.tag, a.arch, a.encoder, a.epochs, a.batch, a.lr, [int(k) for k in a.folds.split(",")])
    else:
        step_eval(a.tag, a.workers, tuple(float(t) for t in a.thresholds.split(",")))


if __name__ == "__main__":
    main()
