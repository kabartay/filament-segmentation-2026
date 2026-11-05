"""Multi-threshold candidates: a component tree with the learned scorer and an exact antichain selection.

The single-threshold pipeline takes the components of the probability map at 0.45 as candidates. A
filament whose best mask comes from another threshold (thinner at 0.65, or split from a neighbour it is
merged with at 0.45) cannot be chosen. Here the candidates are the components at several thresholds
LEVELS. Thresholding and the grouping closing are both monotone, so every component at a higher level
lies inside exactly one component at the next lower level: the candidates form a forest.

The scorer regresses the same target as ``filseg.scorer`` (matched IoU per annotator entry, Section
"Component scorer" of the report) on the same features plus the level. Pooled PQ is a ratio, but kept
candidates add about y to its numerator and 1/2 to its denominator per entry, so maximising
sum(y_hat - tau) over a set of disjoint candidates is the right objective for tau near PQ / 2. On a forest
the best disjoint set (an antichain) is exact by dynamic programming,
    f(node) = max(y_hat(node) - tau, sum of f(child)),
and an antichain is disjoint by construction.

Steps:
  build:  ``FILSEG_CACHE=data/cache1536 PYTHONPATH=. .venv/bin/python scripts/experiments/component_tree.py build``
          writes runs/oof_tree_1536_v4.pkl (one record per OOF image: features, level, parent, IoU per entry);
  eval:   ``PYTHONPATH=. .venv/bin/python scripts/experiments/component_tree.py eval [--context] [--modes hurdle]``
          cross-validated scorer (trained on the other four folds), PQ for a range of tau, and the
          single-level control (level 0.45 only) that must reproduce the submitted pipeline.
"""

from __future__ import annotations

import os

# one OpenMP thread per process: on a shared, saturated machine scikit-learn's default (all cores)
# spends its time yielding between threads (observed 2026-11-05: an 8-image fit stalled for 10 min)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

from filseg.cache import CACHE
from filseg.data import TRAIN_IMAGES, group_by_stem, load_train, read_image
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import components, upsample
from filseg.scorer import CANDIDATE_CLOSE_PX, CANDIDATE_MIN_AREA, component_features, make_regressor
from filseg.train import RUNS

LEVELS = (0.30, 0.38, 0.45, 0.55, 0.65, 0.75)
BASE_LEVEL = 0.45
NAME = "unet_r34_1536"
PROBS = "val_probs_v4.npy"
OUT = RUNS / "oof_tree_1536_v4.pkl"

G: dict = {}


def init(levels: tuple[float, ...], spread: bool) -> None:
    G["levels"] = levels
    G["groups"] = group_by_stem(load_train())
    G["index"] = json.loads((CACHE / "index.json").read_text())["train"]
    G["images"] = np.load(CACHE / "train_images.npy", mmap_mode="r")
    G["probs"] = {f: np.load(RUNS / f"{NAME}_f{f}" / PROBS, mmap_mode="r") for f in range(5)}
    G["spread"] = ({f: np.load(RUNS / f"{NAME}_f{f}" / "val_spread_v4.npy", mmap_mode="r") for f in range(5)}
                   if spread else None)


def tree_candidates(full: np.ndarray, img: np.ndarray, disk, levels: tuple[float, ...] = LEVELS,
                    spread: np.ndarray | None = None) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    """Candidates at every level: RLEs, features, level and parent index (-1 for roots).

    Args:
        full: (2048, 2048) float32 probability map.
        img: (2048, 2048) uint8 raw image.
        disk: fitted solar disk.
        levels: thresholds, increasing.
        spread: optional (2048, 2048) per-pixel std of the four flipped views; adds uncertainty features.
    """
    rles, feats, level, parent = [], [], [], []
    prev_labels = None  # index of the node covering each pixel at the previous level, -1 outside
    mean_prob = float(full.mean())
    for lv in levels:
        comps = components(full, lv, CANDIDATE_CLOSE_PX, disk, min_area=CANDIDATE_MIN_AREA)
        labels = np.full(full.shape, -1, np.int32)
        for c in comps:
            mask = mask_util.decode(c.rle).astype(bool)
            idx = len(rles)
            labels[mask] = idx
            ft = component_features(mask, full, img, disk, c)
            ft.update(n_comp_image=len(comps), img_mean_prob=mean_prob, level=lv)
            if spread is not None:
                sp = spread[mask]
                ft.update(spread_mean=float(sp.mean()), spread_q90=float(np.quantile(sp, 0.9)),
                          spread_max=float(sp.max()), spread_rel=float(sp.mean() / max(ft["mean"], 1e-6)))
            if prev_labels is None:
                par = -1
            else:
                ys, xs = np.nonzero(mask)
                par = int(prev_labels[ys[0], xs[0]])
                assert par >= 0, "a higher-level component must lie inside a lower-level one"
            ft["parent_area_ratio"] = c.area / feats[par]["area"] if par >= 0 else 1.0
            rles.append(c.rle)
            feats.append(ft)
            level.append(lv)
            parent.append(par)
        prev_labels = labels
    return rles, feats, np.array(level), np.array(parent, dtype=np.int64)


def work(item: tuple[int, int, str]) -> dict:
    f, i, s = item
    full = upsample(np.asarray(G["probs"][f][i], np.float32))
    disk = fit_disk(np.array(G["images"][G["index"].index(s)]))
    spread = upsample(np.asarray(G["spread"][f][i], np.float32)) if G["spread"] is not None else None
    rles, feats, level, parent = tree_candidates(full, read_image(TRAIN_IMAGES / f"{s}.jpeg"), disk,
                                                 G["levels"], spread)
    entries = [(len(e.rles), iou_matrix(e.rles, rles)) for e in G["groups"][s]]
    return dict(fold=f, stem=s, feats=feats, level=level, parent=parent, entries=entries)


def build(limit: int, workers: int, out: Path, levels: tuple[float, ...] = LEVELS, spread: bool = False) -> None:
    items = [(f, i, s) for f in range(5)
             for i, s in enumerate(json.loads((RUNS / f"{NAME}_f{f}" / "val_stems.json").read_text()))]
    if limit:
        items = items[:limit]
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=init, initargs=(levels, spread)) as ex:
        data = list(ex.map(work, items, chunksize=2))
    pickle.dump(data, open(out, "wb"))
    n = sum(len(d["feats"]) for d in data)
    print(f"{len(data)} images, {n} nodes, {time.time() - t0:.0f} s -> {out}")


def table(data: list[dict], names: list[str] | None = None):
    """Features, target and fold of every node (target as in ``filseg.scorer.training_table``)."""
    names = names or sorted(next(d for d in data if d["feats"])["feats"][0])
    X, y, fold, weight = [], [], [], []
    for d in data:
        n = len(d["entries"])
        for j, ft in enumerate(d["feats"]):
            X.append([ft[c] for c in names])
            tot = 0.0
            for n_gt, m in d["entries"]:
                if n_gt and m.shape[1]:
                    col = m[:, j]
                    tot += float(col[col > 0.5].sum())
            y.append(tot / n)
            fold.append(d["fold"])
            weight.append(n)
    return names, np.array(X), np.array(y), np.array(fold), np.array(weight, float)


def antichain(score: np.ndarray, parent: np.ndarray, tau: float) -> list[int]:
    """Disjoint node set maximising sum(score - tau) on the forest given by ``parent``."""
    n = len(score)
    children: list[list[int]] = [[] for _ in range(n)]
    for j, p in enumerate(parent):
        if p >= 0:
            children[p].append(j)
    best = np.zeros(n)
    take = np.zeros(n, bool)
    for j in range(n - 1, -1, -1):  # children always have larger indices than their parent
        below = sum(best[c] for c in children[j])
        own = score[j] - tau
        take[j] = own > 0 and own >= below
        best[j] = max(own, below, 0.0)
    out, stack = [], [j for j in range(n) if parent[j] < 0]
    while stack:
        j = stack.pop()
        if take[j]:
            out.append(j)
        else:
            stack.extend(children[j])
    return out


def pooled(data: list[dict], picks: list[list[int]]) -> PQCounts:
    total = PQCounts()
    for d, kept in zip(data, picks):
        kept = np.array(kept, dtype=np.int64)
        for n_gt, iou in d["entries"]:
            total.add(iou[:, kept] if iou.shape[1] else iou, n_gt, len(kept))
    return total


def make_classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def fit_predict(mode: str, X: np.ndarray, y: np.ndarray, w: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """Scores of ``Xt`` from a model fitted on (X, y).

    mode "plain": regress y (as filseg.scorer); "weighted": the same with rows weighted by the image's
    number of annotator entries, as pooled PQ weights them; "hurdle": P(y > 0) times E[y given y > 0],
    both weighted.
    """
    if mode == "plain":
        return make_regressor().fit(X, y).predict(Xt)
    if mode == "weighted":
        return make_regressor().fit(X, y, sample_weight=w).predict(Xt)
    pos = y > 0
    h = make_classifier().fit(X, pos, sample_weight=w).predict_proba(Xt)[:, 1]
    m = make_regressor().fit(X[pos], y[pos], sample_weight=w[pos]).predict(Xt)
    return h * m


def cv_scores(data: list[dict], keep_node, mode: str = "plain") -> tuple[list[np.ndarray], list[str]]:
    """Per-image node scores from models trained on the other four folds, on nodes ``keep_node``."""
    sub = [dict(d, feats=[ft for ft in d["feats"] if keep_node(ft)]) for d in data]
    names, X, y, fold, w = table(sub)
    score = np.zeros_like(y)
    for f in np.unique(fold):
        score[fold == f] = fit_predict(mode, X[fold != f], y[fold != f], w[fold != f], X[fold == f])
    out, start = [], 0
    for d in sub:
        out.append(score[start:start + len(d["feats"])])
        start += len(d["feats"])
    return out, names


def add_context(data: list[dict]) -> None:
    """Tree-context features, in place: how a node relates to its parent, children and siblings.

    n_children / child_area_frac / max_child_frac: does the node break up at the next level (a merge of
    several filaments, or a filament with weak parts)? n_siblings: how many pieces the parent splits into.
    stable_up / stable_down: number of consecutive levels above / below over which the node keeps at least
    85% of its area (a component that persists across thresholds has a sharp edge).
    """
    for d in data:
        feats, parent = d["feats"], d["parent"]
        n = len(feats)
        children: list[list[int]] = [[] for _ in range(n)]
        for j, p in enumerate(parent):
            if p >= 0:
                children[p].append(j)
        area = np.array([ft["area"] for ft in feats], float)
        up = np.zeros(n)
        for j in range(n):  # parents come first, so their value is ready
            p = parent[j]
            up[j] = up[p] + 1 if p >= 0 and area[j] >= 0.85 * area[p] else 0
        down = np.zeros(n)
        for j in range(n - 1, -1, -1):
            big = [c for c in children[j] if area[c] >= 0.85 * area[j]]
            down[j] = down[big[0]] + 1 if big else 0
        for j, ft in enumerate(feats):
            ch = children[j]
            ft["n_children"] = len(ch)
            ft["child_area_frac"] = float(area[ch].sum() / area[j]) if ch else 0.0
            ft["max_child_frac"] = float(area[ch].max() / area[j]) if ch else 0.0
            ft["n_siblings"] = len(children[parent[j]]) - 1 if parent[j] >= 0 else 0
            ft["stable_up"] = float(up[j])
            ft["stable_down"] = float(down[j])


def image_counts(data: list[dict], picks: list[list[int]]) -> np.ndarray:
    """(images, 4) rows of (matched IoU sum, TP, FP, FN) for a selection."""
    rows = []
    for d, kept in zip(data, picks):
        c = pooled([d], [kept])
        rows.append([c.tp_iou_sum, c.tp, c.fp, c.fn])
    return np.array(rows)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000) -> dict:
    """PQ difference a - b with a 95% interval, resampling images."""
    pq = lambda m: m[..., 0].sum(-1) / (m[..., 1].sum(-1) + 0.5 * m[..., 2].sum(-1) + 0.5 * m[..., 3].sum(-1))
    idx = np.random.default_rng(0).integers(0, len(a), size=(n, len(a)))
    diff = pq(a[idx]) - pq(b[idx])
    return dict(diff=float(pq(a) - pq(b)), ci95=[float(v) for v in np.percentile(diff, [2.5, 97.5])],
                p_le_0=float((diff <= 0).mean()))


def evaluate(path: Path, context: bool = False, modes: tuple[str, ...] = ("plain", "weighted", "hurdle"),
             taus: tuple[float, ...] = (0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28), drop: tuple[str, ...] = (),
             tag: str = "") -> None:
    data = pickle.load(open(path, "rb"))
    if drop:  # ablation: remove features whose name starts with any of these prefixes
        for d in data:
            d["feats"] = [{k: v for k, v in ft.items() if not k.startswith(drop)} for ft in d["feats"]]
    levels = sorted({float(v) for d in data for v in d["level"]})
    if context:
        add_context(data)
    # control: level 0.45 only, original 20 features, plain threshold -- must reproduce the submitted pipeline
    base_names = None
    idx = [np.nonzero(d["level"] == BASE_LEVEL)[0] for d in data]
    tree_only = {"level", "parent_area_ratio", "spread_mean", "spread_q90", "spread_max", "spread_rel", "n_children", "child_area_frac", "max_child_frac", "n_siblings",
                 "stable_up", "stable_down"}
    base = [dict(d, feats=[{k: v for k, v in d["feats"][j].items() if k not in tree_only}
                           for j in ix],
                 entries=[(n, m[:, ix] if m.shape[1] else m) for n, m in d["entries"]]) for d, ix in zip(data, idx)]
    s_base, base_names = cv_scores(base, lambda ft: True)
    results = {"control_level_045": {}}
    control = image_counts(base, [list(np.nonzero(s > 0.20)[0]) for s in s_base])
    for tau in (0.18, 0.20, 0.22):
        c = pooled(base, [list(np.nonzero(s > tau)[0]) for s in s_base])
        results["control_level_045"][tau] = c.score()
        print(f"control (level 0.45 only), tau {tau:.2f}: {c.summary()}")
    for mode in modes:
        s_all, names = cv_scores(data, lambda ft: True, mode)
        key = f"tree_{mode}" + ("_context" if context else "") + (f"_{tag}" if tag else "")
        results[key] = {}
        for tau in taus:
            picks = [antichain(s, d["parent"], tau) for s, d in zip(s_all, data)]
            c = pooled(data, picks)
            sq = c.tp_iou_sum / max(c.tp, 1)
            per_fold = {int(f): pooled([d for d in data if d["fold"] == f],
                                       [p for p, d in zip(picks, data) if d["fold"] == f]).score() for f in range(5)}
            levels = {str(lv): int(sum(int((d["level"][np.array(p, dtype=np.int64)] == lv).sum())
                                       for p, d in zip(picks, data))) for lv in levels}
            results[key][tau] = dict(pq=c.score(), sq=sq, tp=c.tp, fp=c.fp, fn=c.fn, per_fold=per_fold, levels=levels)
            results[key][tau]["vs_control"] = paired_bootstrap(image_counts(data, picks), control)
            print(f"{key}, tau {tau:.2f}: {c.summary()}  per fold "
                  f"{ {k: round(v, 4) for k, v in per_fold.items()} }  vs control {results[key][tau]['vs_control']}",
                  flush=True)
    out = path.with_name(path.stem + ("_context" if context else "") + (f"_{tag}" if tag else "") + ".json")
    out.write_text(json.dumps(results, indent=1, default=float))
    print("features:", names, "->", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "eval"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--context", action="store_true", help="add tree-context features")
    ap.add_argument("--levels", default=",".join(str(v) for v in LEVELS), help="build: thresholds, increasing")
    ap.add_argument("--spread", action="store_true", help="build: add view-spread features (val_spread_v4.npy)")
    ap.add_argument("--drop", default="", help="eval: comma-separated feature-name prefixes to leave out")
    ap.add_argument("--tag", default="", help="eval: suffix for result keys and the output json")
    ap.add_argument("--modes", default="plain,weighted,hurdle")
    ap.add_argument("--taus", default="0.16,0.18,0.20,0.22,0.24,0.26,0.28")
    a = ap.parse_args()
    if a.step == "build":
        build(a.limit, a.workers, a.out, tuple(float(v) for v in a.levels.split(",")), a.spread)
    else:
        evaluate(a.out, a.context, tuple(a.modes.split(",")), tuple(float(t) for t in a.taus.split(",")),
                 tuple(x for x in a.drop.split(",") if x), a.tag)


if __name__ == "__main__":
    main()
