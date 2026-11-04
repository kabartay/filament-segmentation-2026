"""Second-stage component scorer: keep components by their predicted PQ contribution.

The hand rule (area >= 300 px and mean probability >= 0.8) treats every component alike. A kept
component that matches annotator entries adds the sum of those IoUs to the PQ numerator, and costs
about half a point in the denominator either way, so it pays when its expected matched IoU per entry
is above about PQ / 2. A gradient-boosted regressor learns that expectation,

    y = sum over the image's entries of IoU * [IoU > 0.5] / number of entries,

from shape, probability, contrast and disk-position features of out-of-fold candidates. At inference
we keep the candidates with predicted y above ``tau``. On the 5-fold OOF this gives PQ 0.4162 against
0.4086 for the rule, and it wins on every fold (docs/DESIGN.md, "Second-stage component scorer").

Fit: ``.venv/bin/python -m filseg.scorer --components runs/oof_components.pkl --out weights/component_scorer.pkl``
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

from filseg.data import SIZE
from filseg.geometry import Disk
from filseg.postprocess import Component, components

CANDIDATE_THRESHOLD = 0.45
CANDIDATE_CLOSE_PX = 31
CANDIDATE_MIN_AREA = 100
DEFAULT_TAU = 0.22


def component_features(mask: np.ndarray, full: np.ndarray, img: np.ndarray, disk: Disk,
                       comp: Component) -> dict[str, float]:
    """Per-component features.

    Args:
        mask: boolean (2048, 2048) mask of the component.
        full: (2048, 2048) float32 probability map.
        img: (2048, 2048) uint8 raw H-alpha image.
        disk: fitted solar disk.
        comp: the component's area, mean probability and longitude.
    """
    ys, xs = np.nonzero(mask)
    p = full[ys, xs]
    cy, cx = ys.mean(), xs.mean()
    cov = np.cov(np.stack([xs, ys])) if len(xs) > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    elong = float(np.sqrt(ev[0] / max(ev[1], 1e-6)))
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = 25
    X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, SIZE), min(y1 + pad, SIZE)
    sub_m = mask[Y0:Y1, X0:X1].astype(np.uint8)
    sub_i = img[Y0:Y1, X0:X1].astype(np.float32)
    ring = cv2.dilate(sub_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) & (1 - sub_m)
    inside = sub_i[sub_m > 0].mean()
    around = sub_i[ring > 0].mean() if ring.any() else inside
    contour = max(cv2.findContours(sub_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)
    perim = cv2.arcLength(contour, True)
    rr = np.hypot(cx - disk.cx, cy - disk.cy) / disk.r
    length = float(4 * np.sqrt(ev[0]))
    return dict(area=comp.area, log_area=float(np.log(comp.area)), mean=comp.score, max=float(p.max()),
                std=float(p.std()), q10=float(np.quantile(p, 0.1)), q50=float(np.median(p)),
                q90=float(np.quantile(p, 0.9)), frac_hi=float((p > 0.8).mean()), elong=elong,
                length=length, width=float(comp.area / max(length, 1)),
                compact=float(4 * np.pi * comp.area / max(perim, 1) ** 2),
                contrast=float((around - inside) / max(around, 1)), inside=float(inside), r_disk=float(rr),
                lon=float(comp.lon),
                lat=float(np.degrees(np.arcsin(np.clip((disk.cy - cy) / disk.r, -1, 1)))))


def candidates(full: np.ndarray, img: np.ndarray, disk: Disk) -> tuple[list[Component], list[dict]]:
    """Candidate components of one image and their features (image-level ones included)."""
    comps = components(full, CANDIDATE_THRESHOLD, CANDIDATE_CLOSE_PX, disk, min_area=CANDIDATE_MIN_AREA)
    feats = [component_features(mask_util.decode(c.rle).astype(bool), full, img, disk, c) for c in comps]
    mean_prob = float(full.mean())
    for ft in feats:
        ft["n_comp_image"] = len(comps)
        ft["img_mean_prob"] = mean_prob
    return comps, feats


@dataclass
class Scorer:
    """A fitted regressor, its feature order and the keep threshold."""

    model: object
    names: list[str]
    tau: float = DEFAULT_TAU

    def predict(self, feats: list[dict]) -> np.ndarray:
        if not feats:
            return np.zeros(0)
        return self.model.predict(np.array([[ft[c] for c in self.names] for ft in feats]))

    def select(self, full: np.ndarray, img: np.ndarray, disk: Disk) -> list[dict]:
        """Instance RLEs kept for one image (full: 2048 px probability map)."""
        comps, feats = candidates(full, img, disk)
        score = self.predict(feats)
        return [c.rle for c, s in zip(comps, score) if s > self.tau]


def load_scorer(path: Path) -> Scorer:
    """Load a scorer saved by ``main`` (a dict holding the regressor, feature names and tau)."""
    with open(path, "rb") as handle:
        saved = pickle.load(handle)
    return Scorer(saved["model"], saved["names"], saved["tau"])


def make_regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def training_table(data: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Features X, target y and fold of every cached OOF candidate (see component_features.py)."""
    names = sorted(next(d for d in data if d["feats"])["feats"][0])
    X, y, fold = [], [], []
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
    return names, np.array(X), np.array(y), np.array(fold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = parser.parse_args()
    names, X, y, _ = training_table(pickle.load(open(args.components, "rb")))
    model = make_regressor().fit(X, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as handle:  # a plain dict, so loading does not depend on how this module was run
        pickle.dump({"model": model, "names": names, "tau": args.tau}, handle)
    print(f"fitted on {len(y)} candidates, {len(names)} features, tau {args.tau} -> {args.out}")


if __name__ == "__main__":
    main()
