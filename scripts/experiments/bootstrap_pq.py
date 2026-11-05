"""Uncertainty of the out-of-fold PQ: per-fold spread and a bootstrap over images.

PQ is a ratio of sums pooled over annotator-image entries, so we resample whole images (all entries of an
image together) with replacement and recompute the pooled ratio. The paired bootstrap of the difference
resamples the same images for both pipelines.

Pipelines compared:
  final  -- 1536 px, 4 flips, component scorer (cross-validated, tau 0.20), from the cached OOF candidates;
  rule   -- 1024 px, 2 views, hand rule of submission 4 (threshold 0.45, closing 31 px, area >= 300,
            confidence >= 0.8), from the fold models' saved OOF probabilities.

Run (FILSEG_CACHE unset, so the 1024 px cache is used for the rule pipeline):
``.venv/bin/python scripts/experiments/bootstrap_pq.py``
Writes runs/bootstrap_pq.json.
"""

from __future__ import annotations

import os

# one OpenMP thread per process: on a shared, saturated machine scikit-learn's default (all cores)
# spends its time yielding between threads (observed 2026-11-05: an 8-image fit stalled for 10 min)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import pickle
from pathlib import Path

import numpy as np

from filseg.cache import CACHE
from filseg.data import ROOT, group_by_stem, load_train
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import PostConfig, instances
from filseg.scorer import make_regressor, training_table
from filseg.train import RUNS

COMPONENTS = RUNS / "oof_components_1536_v4.pkl"
TAU = 0.20
RULE = PostConfig(threshold=0.45, close_px=31, min_area=300, min_score=0.8, max_lon=90.0)
N_BOOT = 2000


def counts_row(c: PQCounts) -> list[float]:
    """(sum of matched IoU, TP, FP, FN) of one image."""
    return [c.tp_iou_sum, c.tp, c.fp, c.fn]


def final_per_image() -> tuple[dict[str, list[float]], dict[str, int]]:
    """Per-image counts and fold of the final pipeline (scorer trained on the other four folds)."""
    data = pickle.load(open(COMPONENTS, "rb"))
    _, X, y, fold = training_table(data)
    score = np.zeros_like(y)
    for f in np.unique(fold):
        score[fold == f] = make_regressor().fit(X[fold != f], y[fold != f]).predict(X[fold == f])
    rows, folds, start = {}, {}, 0
    for d in data:
        n = len(d["feats"])
        kept = np.nonzero(score[start:start + n] > TAU)[0]
        start += n
        c = PQCounts()
        for n_gt, iou in d["entries"]:
            c.add(iou[:, kept] if iou.shape[1] else iou, n_gt, len(kept))
        rows[d["stem"]], folds[d["stem"]] = counts_row(c), d["fold"]
    return rows, folds


def rule_per_image() -> dict[str, list[float]]:
    """Per-image counts of the 1024 px hand-rule pipeline on the five fold models' OOF probabilities."""
    index = json.loads((CACHE / "index.json").read_text())["train"]
    images = np.load(CACHE / "train_images.npy", mmap_mode="r")
    groups = group_by_stem(load_train())
    rows = {}
    for k in range(5):
        run = RUNS / f"unet_r34full_f{k}"
        stems = json.loads((run / "val_stems.json").read_text())
        probs = np.load(run / "val_probs.npy", mmap_mode="r")
        for prob, stem in zip(probs, stems):
            disk = fit_disk(np.array(images[index.index(stem)]))
            pred = instances(np.asarray(prob, np.float32), RULE, disk)
            c = PQCounts()
            for entry in groups[stem]:
                c.add(iou_matrix(entry.rles, pred), len(entry.rles), len(pred))
            rows[stem] = counts_row(c)
    return rows


def pq(m: np.ndarray) -> np.ndarray:
    """Pooled PQ of count rows; ``m`` is (..., images, 4)."""
    s = m.sum(axis=-2)
    return s[..., 0] / (s[..., 1] + 0.5 * s[..., 2] + 0.5 * s[..., 3])


def main() -> None:
    final, folds = final_per_image()
    rule = rule_per_image()
    stems = sorted(final)
    assert set(stems) == set(rule), "the two pipelines must cover the same images"
    A = np.array([final[s] for s in stems])
    B = np.array([rule[s] for s in stems])
    fold = np.array([folds[s] for s in stems])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(stems), size=(N_BOOT, len(stems)))
    boot_a, boot_b = pq(A[idx]), pq(B[idx])
    diff = boot_a - boot_b
    per_fold = {int(f): float(pq(A[fold == f])) for f in np.unique(fold)}
    per_fold_rule = {int(f): float(pq(B[fold == f])) for f in np.unique(fold)}
    out = {
        "images": len(stems),
        "final_pq": float(pq(A)), "rule_pq": float(pq(B)),
        "final_ci95": [float(v) for v in np.percentile(boot_a, [2.5, 97.5])],
        "rule_ci95": [float(v) for v in np.percentile(boot_b, [2.5, 97.5])],
        "final_boot_std": float(boot_a.std()), "rule_boot_std": float(boot_b.std()),
        "diff": float(pq(A) - pq(B)),
        "diff_ci95": [float(v) for v in np.percentile(diff, [2.5, 97.5])],
        "diff_p_le_0": float((diff <= 0).mean()),
        "final_per_fold": per_fold, "rule_per_fold": per_fold_rule,
        "final_fold_mean": float(np.mean(list(per_fold.values()))),
        "final_fold_std": float(np.std(list(per_fold.values()), ddof=1)),
        "n_boot": N_BOOT,
    }
    path = RUNS / "bootstrap_pq.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
