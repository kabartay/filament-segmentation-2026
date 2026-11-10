"""Scorer-feature variants on the cached tree nodes (explorer ``adopt-scorer-features``).

Ideas from a public notebook (tushhaaarrrr): lower scorer capacity (15 leaves), the observing station as a
feature, neighbour features (count / area / probability of nodes of other tree branches within a radius).

Steps:
  centroids: node centroids (x, y px at 2048) of the cached tree, recomputed from the OOF probabilities with
             ``postprocess.components`` only (no features); checked against the cached node areas.
             Writes runs/adopt-scorer-features_centroids.pkl.
  eval:      cross-validated hurdle per variant (scorer fitted on the other four folds), antichain selection at
             several tau, per-fold PQ and a paired bootstrap over images against the baseline selection.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pycocotools import mask as mask_util

from filseg.cache import CACHE
from filseg.geometry import fit_disk
from filseg.metric import PQCounts
from filseg.postprocess import components, upsample
from filseg.scorer import CANDIDATE_CLOSE_PX, CANDIDATE_MIN_AREA, make_regressor
from filseg.tree import TREE_LEVELS, add_context, antichain
from filseg.train import RUNS

NODES = RUNS / "oof_tree12s_1536_v4.pkl"
CENT = RUNS / "adopt-scorer-features_centroids.pkl"
NAME = "unet_r34_1536"
STATIONS = ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")
G: dict = {}


def _init() -> None:
    G["index"] = json.loads((CACHE / "index.json").read_text())["train"]
    G["images"] = np.load(CACHE / "train_images.npy", mmap_mode="r")
    G["probs"] = {f: np.load(RUNS / f"{NAME}_f{f}" / "val_probs_v4.npy", mmap_mode="r") for f in range(5)}
    G["stems"] = {f: json.loads((RUNS / f"{NAME}_f{f}" / "val_stems.json").read_text()) for f in range(5)}


def _centroids(item: tuple[int, str]) -> tuple[str, np.ndarray]:
    f, s = item
    i = G["stems"][f].index(s)
    full = upsample(np.asarray(G["probs"][f][i], np.float32))
    disk = fit_disk(np.array(G["images"][G["index"].index(s)]))
    rows = []
    for lv in TREE_LEVELS:
        for c in components(full, lv, CANDIDATE_CLOSE_PX, disk, min_area=CANDIDATE_MIN_AREA):
            ys, xs = np.nonzero(mask_util.decode(c.rle))
            rows.append([xs.mean(), ys.mean(), c.area, disk.cx, disk.cy, disk.r])
    return s, np.array(rows).reshape(-1, 6)


def build_centroids(workers: int, limit: int = 0) -> None:
    """Centroid sidecar of the cached tree nodes (same node order as the cache)."""
    data = pickle.load(open(NODES, "rb"))[: limit or None]
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        res = dict(ex.map(_centroids, [(d["fold"], d["stem"]) for d in data], chunksize=2))
    bad = [d["stem"] for d in data
           if len(res[d["stem"]]) != len(d["feats"])
           or not np.allclose(res[d["stem"]][:, 2], [ft["area"] for ft in d["feats"]])]
    print(f"{len(res)} images, area mismatches: {len(bad)} {bad[:5]}")
    pickle.dump(res, open(CENT if not limit else CENT.with_suffix(".smoke.pkl"), "wb"))


# ---------------------------------------------------------------- features

def add_station(data: list[dict]) -> None:
    """One-hot observing station from the image stem's trailing code (e.g. ``20110120105534Ch``)."""
    for d in data:
        code = d["stem"][-2:]
        for ft in d["feats"]:
            for k in STATIONS:
                ft[f"st_{k}"] = float(code == k)


def add_neighbours(data: list[dict], cent: dict, radii: tuple[float, ...] = (150.0, 400.0)) -> None:
    """Nodes of other tree branches near each node (centroid distance, px at 2048).

    A node's own branch is its tree root's subtree. Per radius: count, summed area and max mean probability of
    the same-level nodes of other branches; plus the distance to the nearest one.
    """
    for d in data:
        feats, par = d["feats"], d["parent"]
        if not feats:
            continue
        xy = cent[d["stem"]][:, :2]
        n = len(feats)
        root = np.arange(n)
        for j in range(n):
            if par[j] >= 0:
                root[j] = root[par[j]]
        lev = d["level"]
        area = np.array([ft["area"] for ft in feats])
        prob = np.array([ft["mean"] for ft in feats])
        dist = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        other = (root[:, None] != root[None]) & (lev[:, None] == lev[None])
        for j, ft in enumerate(feats):
            dj = np.where(other[j], dist[j], np.inf)
            ft["nb_nearest"] = float(min(dj.min(), 1000.0))
            for r in radii:
                sel = dj < r
                ft[f"nb_n_{int(r)}"] = float(sel.sum())
                ft[f"nb_area_{int(r)}"] = float(area[sel].sum())
                ft[f"nb_pmax_{int(r)}"] = float(prob[sel].max()) if sel.any() else 0.0


# ---------------------------------------------------------------- evaluation

def table(data: list[dict], names: list[str]):
    """Features, target, fold and weight of every node (as ``filseg.tree.training_table``)."""
    X, y, fold, w = [], [], [], []
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
            w.append(n)
    return np.array(X), np.array(y), np.array(fold), np.array(w, float)


def cv_scores(data: list[dict], names: list[str], leaves: int = 31, reg_leaves: int | None = None) -> list[np.ndarray]:
    """Per-image node scores of a fold-held-out hurdle; classifier with ``leaves``, regressor with ``reg_leaves``."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    X, y, fold, w = table(data, names)
    rp = make_regressor().get_params()
    if reg_leaves is not None or leaves != 31:
        rp["max_leaf_nodes"] = reg_leaves if reg_leaves is not None else leaves
    hp = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=leaves, min_samples_leaf=40,
              l2_regularization=1.0, random_state=0)
    score = np.zeros_like(y)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        h = HistGradientBoostingClassifier(**hp).fit(X[tr], y[tr] > 0, sample_weight=w[tr])
        pos = tr & (y > 0)
        m = HistGradientBoostingRegressor(**rp).fit(X[pos], y[pos], sample_weight=w[pos])
        score[te] = h.predict_proba(X[te])[:, 1] * m.predict(X[te])
    out, s = [], 0
    for d in data:
        out.append(score[s:s + len(d["feats"])])
        s += len(d["feats"])
    return out


def per_image(data: list[dict], scores: list[np.ndarray], tau: float) -> np.ndarray:
    """(images, 4) rows of (matched IoU sum, TP, FP, FN) of the antichain selection."""
    rows = []
    for d, s in zip(data, scores):
        kept = np.array(antichain(s, d["parent"], tau), dtype=np.int64)
        c = PQCounts()
        for n_gt, iou in d["entries"]:
            c.add(iou[:, kept] if iou.shape[1] else iou, n_gt, len(kept))
        rows.append([c.tp_iou_sum, c.tp, c.fp, c.fn])
    return np.array(rows)


def pq(m: np.ndarray) -> np.ndarray:
    """Pooled PQ of count rows, shape (..., images, 4)."""
    s = m.sum(axis=-2)
    return s[..., 0] / (s[..., 1] + 0.5 * s[..., 2] + 0.5 * s[..., 3])


def compare(a: np.ndarray, b: np.ndarray, fold: np.ndarray, n: int = 2000) -> dict:
    """PQ of a, a - b with a paired bootstrap over images, and the per-fold differences."""
    idx = np.random.default_rng(0).integers(0, len(a), size=(n, len(a)))
    diff = pq(a[idx]) - pq(b[idx])
    pf = [float(pq(a[fold == f]) - pq(b[fold == f])) for f in range(5)]
    return dict(pq=float(pq(a)), diff=float(pq(a) - pq(b)), ci95=[float(v) for v in np.percentile(diff, [2.5, 97.5])],
                p_le_0=float((diff <= 0).mean()), per_fold_diff=pf, folds_won=int(sum(v > 0 for v in pf)))


def evaluate(taus: tuple[float, ...], nb_off: bool = False) -> None:
    data = pickle.load(open(NODES, "rb"))
    for d in data:
        add_context(d["feats"], d["parent"])
    base_names = sorted(data[0]["feats"][0])
    add_station(data)
    if not nb_off:
        add_neighbours(data, pickle.load(open(CENT, "rb")))
    fold = np.array([d["fold"] for d in data])
    st = [f"st_{k}" for k in STATIONS]
    nb = [k for k in sorted(data[0]["feats"][0]) if k.startswith("nb_")]
    tag = "_nonb" if nb_off else ""
    variants = {
        "base": (base_names, 31, None),
        "leaves15": (base_names, 15, None),
        "leaves15_reg31": (base_names, 15, 31),
        "leaves7": (base_names, 7, None),
        "station": (base_names + st, 31, None),
        "neighbours": (base_names + nb, 31, None),
        "station+nb": (base_names + st + nb, 31, None),
        "leaves15+station": (base_names + st, 15, None),
        "leaves15+nb": (base_names + nb, 15, None),
        "leaves15+station+nb": (base_names + st + nb, 15, None),
    }
    if nb_off:
        variants = {k: v for k, v in variants.items() if "nb" not in k and "neighbours" not in k}
    else:  # the no-neighbour variants are in the --no-nb json; keep only the baseline and neighbour ones
        variants = {k: v for k, v in variants.items() if k == "base" or "nb" in k or "neighbours" in k}
    pi = {}
    for name, (names, lv, rl) in variants.items():
        sc = cv_scores(data, names, lv, rl)
        pi[name] = {t: per_image(data, sc, t) for t in taus}
        print(name, {t: round(float(pq(pi[name][t])), 4) for t in taus}, flush=True)
    out: dict = {"n_variants_tried": len(variants) - 1, "taus": taus}
    for name in variants:
        if name == "base":
            continue
        out[name] = {str(t): compare(pi[name][t], pi["base"][t], fold) for t in taus}
        for t in taus:
            r = out[name][str(t)]
            print(f"{name:22s} tau {t:.2f}: {r['pq']:.4f} d {r['diff']:+.4f} {[round(v, 4) for v in r['ci95']]} "
                  f"p {r['p_le_0']:.3f} folds+ {r['folds_won']}/5 {[round(v, 4) for v in r['per_fold_diff']]}", flush=True)
    out["base"] = {str(t): float(pq(pi["base"][t])) for t in taus}
    (RUNS / f"adopt-scorer-features_eval{tag}.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["centroids", "score"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-nb", action="store_true", help="eval: skip neighbour variants (no centroid sidecar needed)")
    ap.add_argument("--taus", default="0.24,0.26,0.28")
    a = ap.parse_args()
    if a.step == "centroids":
        build_centroids(a.workers, a.limit)
    else:
        evaluate(tuple(float(t) for t in a.taus.split(",")), a.no_nb)


if __name__ == "__main__":
    main()
