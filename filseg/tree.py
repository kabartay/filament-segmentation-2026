"""Instance selection on a component tree over thresholds, with a learned hurdle scorer.

The probability map is thresholded at several levels (``TREE_LEVELS``). Thresholding and the grouping
closing are monotone, so every component at a higher level lies inside exactly one component at the next
lower level, and the candidates form a forest: a filament can be taken whole at a low threshold, thinner at
a high one, or split from a neighbour it merges with lower down.

Each node gets the features of ``filseg.scorer.component_features`` plus its level, tree context (how it
relates to its parent, children and siblings) and the spread of the four flipped views inside it (where the
views disagree, the model is unsure). A hurdle model scores it, P(match) x E[matched IoU per entry | match],
both gradient-boosted and trained on out-of-fold nodes with rows weighted by the image's number of annotator
entries, as pooled PQ weights them. A kept node adds about its score to the PQ numerator and 1/2 per entry to
the denominator, so the submission is the disjoint node set (an antichain of the forest) that maximises
sum(score - tau), solved exactly by dynamic programming:

    f(node) = max(score(node) - tau, sum of f(child)).

Out of fold this gives PQ 0.4267 against 0.4219 for the single-threshold scorer (docs/DESIGN.md, "12-level
tree with view-spread features").

Fit: ``.venv/bin/python -m filseg.tree --nodes runs/oof_tree12s_1536_v4.pkl --out weights/tree_scorer_1536.pkl``
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

from filseg.geometry import Disk
from filseg.postprocess import components
from filseg.scorer import CANDIDATE_CLOSE_PX, CANDIDATE_MIN_AREA, component_features, make_regressor

TREE_LEVELS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
DEFAULT_TAU = 0.26
STABLE_FRACTION = 0.85  # a node "persists" across a level if it keeps at least this share of its area


def tree_candidates(full: np.ndarray, img: np.ndarray, disk: Disk, levels: tuple[float, ...] = TREE_LEVELS,
                    spread: np.ndarray | None = None) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    """Candidates at every level: RLEs, features, level and parent index (-1 for roots).

    Nodes are numbered level by level, so a parent always has a smaller index than its children.

    Args:
        full: (2048, 2048) float32 probability map.
        img: (2048, 2048) uint8 raw H-alpha image.
        disk: fitted solar disk.
        levels: thresholds, increasing.
        spread: optional (2048, 2048) per-pixel std of the four flipped views; adds uncertainty features.

    Returns:
        (rles, feats, level, parent): COCO RLEs, feature dicts, (n,) thresholds and (n,) parent indices.
    """
    rles: list[dict] = []
    feats: list[dict] = []
    level: list[float] = []
    parent: list[int] = []
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


def children_of(parent: np.ndarray) -> list[list[int]]:
    """Child indices of every node of the forest given by ``parent``."""
    children: list[list[int]] = [[] for _ in range(len(parent))]
    for j, p in enumerate(parent):
        if p >= 0:
            children[p].append(j)
    return children


def add_context(feats: list[dict], parent: np.ndarray) -> None:
    """Tree-context features of one image's nodes, in place.

    n_children / child_area_frac / max_child_frac: does the node break up at the next level (a merge of
    several filaments, or a filament with weak parts)? n_siblings: how many pieces the parent splits into.
    stable_up / stable_down: number of consecutive levels above / below over which the node keeps at least
    ``STABLE_FRACTION`` of its area (a component that persists across thresholds has a sharp edge).
    """
    n = len(feats)
    children = children_of(parent)
    area = np.array([ft["area"] for ft in feats], float)
    up = np.zeros(n)
    for j in range(n):  # parents come first, so their value is ready
        p = parent[j]
        up[j] = up[p] + 1 if p >= 0 and area[j] >= STABLE_FRACTION * area[p] else 0
    down = np.zeros(n)
    for j in range(n - 1, -1, -1):
        big = [c for c in children[j] if area[c] >= STABLE_FRACTION * area[j]]
        down[j] = down[big[0]] + 1 if big else 0
    for j, ft in enumerate(feats):
        ch = children[j]
        ft["n_children"] = len(ch)
        ft["child_area_frac"] = float(area[ch].sum() / area[j]) if ch else 0.0
        ft["max_child_frac"] = float(area[ch].max() / area[j]) if ch else 0.0
        ft["n_siblings"] = len(children[parent[j]]) - 1 if parent[j] >= 0 else 0
        ft["stable_up"] = float(up[j])
        ft["stable_down"] = float(down[j])


def antichain(score: np.ndarray, parent: np.ndarray, tau: float) -> list[int]:
    """Disjoint node set maximising sum(score - tau) on the forest given by ``parent`` (exact)."""
    n = len(score)
    children = children_of(parent)
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


def make_classifier():
    """Gradient-boosted P(match) model, with the regressor's hyperparameters."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def fit_hurdle(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[object, object]:
    """Fit P(y > 0) and E[y | y > 0], rows weighted by ``w``."""
    pos = y > 0
    h = make_classifier().fit(X, pos, sample_weight=w)
    m = make_regressor().fit(X[pos], y[pos], sample_weight=w[pos])
    return h, m


@dataclass
class TreeScorer:
    """A fitted hurdle model, its feature order, the keep threshold and the tree levels."""

    h: object
    m: object
    names: list[str]
    tau: float = DEFAULT_TAU
    levels: tuple[float, ...] = TREE_LEVELS

    def predict(self, feats: list[dict]) -> np.ndarray:
        """Expected matched IoU per annotator entry of each node."""
        if not feats:
            return np.zeros(0)
        X = np.array([[ft[c] for c in self.names] for ft in feats])
        return self.h.predict_proba(X)[:, 1] * self.m.predict(X)

    def select(self, full: np.ndarray, img: np.ndarray, disk: Disk, spread: np.ndarray) -> list[dict]:
        """Instance RLEs kept for one image.

        Args:
            full: (2048, 2048) float32 probability map (mean of the four flipped views).
            img: (2048, 2048) uint8 raw image.
            disk: fitted solar disk.
            spread: (2048, 2048) per-pixel std of the four flipped views.
        """
        return self.select_scored(full, img, disk, spread)[0]

    def select_scored(self, full: np.ndarray, img: np.ndarray, disk: Disk,
                      spread: np.ndarray) -> tuple[list[dict], np.ndarray]:
        """``select`` together with the kept instances' scores (stage 2 absorbs the lower-scored first)."""
        rles, feats, _, parent = tree_candidates(full, img, disk, self.levels, spread)
        add_context(feats, parent)
        score = self.predict(feats)
        chosen = sorted(antichain(score, parent, self.tau))
        return [rles[j] for j in chosen], np.asarray(score)[chosen]


def load_tree_scorer(path: Path) -> TreeScorer:
    """Load a scorer saved by ``main`` (a plain dict, so loading does not depend on how it was fitted)."""
    with open(path, "rb") as handle:
        saved = pickle.load(handle)
    return TreeScorer(saved["h"], saved["m"], saved["names"], saved["tau"], tuple(saved["levels"]))


def training_table(data: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Features X, target y, fold and weight of every cached out-of-fold node.

    y is the matched IoU (IoU > 0.5) summed over the image's annotator entries and divided by their number;
    the weight is that number.
    """
    names = sorted(next(d for d in data if d["feats"])["feats"][0])
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=Path, required=True, help="out-of-fold tree nodes (component_tree.py build)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = parser.parse_args()
    data = pickle.load(open(args.nodes, "rb"))
    for d in data:
        add_context(d["feats"], d["parent"])
    levels = sorted({float(v) for d in data for v in d["level"]})
    names, X, y, _, w = training_table(data)
    h, m = fit_hurdle(X, y, w)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump({"h": h, "m": m, "names": names, "tau": args.tau, "levels": levels}, handle)
    print(f"fitted on {len(y)} nodes, {len(names)} features, tau {args.tau}, levels {levels} -> {args.out}")


if __name__ == "__main__":
    main()
