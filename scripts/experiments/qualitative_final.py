"""Qualitative figure of the final pipeline: out-of-fold predictions next to every annotator's masks.

Final pipeline = 1536 px fold models, 4-flip probabilities, candidates from ``filseg.scorer.candidates``,
kept by a component scorer trained on the other four folds (tau 0.20), exactly as in the cached OOF
candidates (runs/oof_components_1536_v4.pkl). Picks fold-0 validation images with three annotators at
the 50th, 75th and 90th percentile of image PQ among images with at least
MIN_GT filaments per annotator (so that every panel shows several filaments); each annotator in its own colour, the prediction in red.

With ``--tree`` the pipeline is the final one instead: tree over 12 thresholds, hurdle scorer with context and
view-spread features trained on the other four folds (tau 0.26), from runs/oof_tree12s_1536_v4.pkl.

Run: ``FILSEG_CACHE=data/cache1536 PYTHONPATH=. .venv/bin/python scripts/experiments/qualitative_final.py [--tree]``
Writes docs/figures/qualitative_final.jpg (``--tree``: qualitative_tree.jpg).
"""

from __future__ import annotations

import argparse
import json
import pickle

import cv2
import numpy as np
from pycocotools import mask as mu

from filseg.cache import CACHE
from filseg.data import ROOT, TRAIN_IMAGES, group_by_stem, load_train, read_image
from filseg.geometry import fit_disk
from filseg.metric import PQCounts
from filseg.postprocess import upsample
from filseg.scorer import candidates, make_regressor, training_table
from filseg.train import RUNS

TAU = 0.20
MIN_GT = 5
FOLD = 0
COLOURS = [(0, 200, 0), (0, 200, 255), (255, 160, 0)]  # BGR, one per annotator
OUT = ROOT / "docs" / "figures" / "qualitative_final.jpg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="store_true", help="draw the tree pipeline")
    a = ap.parse_args()
    if a.tree:
        from filseg.tree import add_context, antichain, fit_hurdle
        from filseg.tree import training_table as tree_table
        data = pickle.load(open(RUNS / "oof_tree12s_1536_v4.pkl", "rb"))
        for d in data:
            add_context(d["feats"], d["parent"])
        _, X, y, fold, w = tree_table(data)
        h, m = fit_hurdle(X[fold != FOLD], y[fold != FOLD], w[fold != FOLD])
        score = h.predict_proba(X[fold == FOLD])[:, 1] * m.predict(X[fold == FOLD])
    else:
        data = pickle.load(open(RUNS / "oof_components_1536_v4.pkl", "rb"))
        _, X, y, fold = training_table(data)
        score = make_regressor().fit(X[fold != FOLD], y[fold != FOLD]).predict(X[fold == FOLD])
    groups = group_by_stem(load_train())
    scored, start = [], 0
    for d in (d for d in data if d["fold"] == FOLD):
        n = len(d["feats"])
        s = score[start:start + n]
        kept = (np.array(antichain(s, d["parent"], 0.26), dtype=np.int64) if a.tree
                else np.nonzero(s > TAU)[0])
        start += n
        if len(d["entries"]) != 3 or np.mean([n_gt for n_gt, _ in d["entries"]]) < MIN_GT:
            continue
        c = PQCounts()
        for n_gt, iou in d["entries"]:
            c.add(iou[:, kept] if iou.shape[1] else iou, n_gt, len(kept))
        scored.append((c.score(), d["stem"], kept, n))
    scored.sort(key=lambda t: t[0])
    picks = [scored[int(q * (len(scored) - 1))] for q in (0.5, 0.75, 0.9)]

    index = json.loads((CACHE / "index.json").read_text())["train"]
    images = np.load(CACHE / "train_images.npy", mmap_mode="r")
    stems = json.loads((RUNS / f"unet_r34_1536_f{FOLD}" / "val_stems.json").read_text())
    probs = np.load(RUNS / f"unet_r34_1536_f{FOLD}" / "val_probs_v4.npy", mmap_mode="r")
    spreads = np.load(RUNS / f"unet_r34_1536_f{FOLD}" / "val_spread_v4.npy", mmap_mode="r") if a.tree else None
    panels = []
    for pq, stem, kept, n in picks:
        raw = read_image(TRAIN_IMAGES / f"{stem}.jpeg")
        full = upsample(np.asarray(probs[stems.index(stem)], np.float32))
        disk = fit_disk(np.array(images[index.index(stem)]))
        if a.tree:
            from filseg.tree import tree_candidates
            sp = upsample(np.asarray(spreads[stems.index(stem)], np.float32))
            rles = tree_candidates(full, raw, disk, spread=sp)[0]
        else:
            rles = [c.rle for c in candidates(full, raw, disk)[0]]
        assert len(rles) == n, f"{stem}: {len(rles)} candidates, cache has {n}"
        img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        for entry, col in zip(groups[stem], COLOURS):
            for r in entry.rles:
                cnts, _ = cv2.findContours(mu.decode(r), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cnts, -1, col, 3)
        for j in kept:
            cnts, _ = cv2.findContours(mu.decode(rles[j]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, (40, 40, 255), 5)
        img = cv2.resize(img[160:1888, 160:1888], (900, 900), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"{stem}  PQ {pq:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        panels.append(img)
    out = OUT.with_name("qualitative_tree.jpg") if a.tree else OUT
    cv2.imwrite(str(out), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print([(round(p, 3), s) for p, s, _, _ in picks], "->", out)


if __name__ == "__main__":
    main()
