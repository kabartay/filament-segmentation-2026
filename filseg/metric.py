"""Panoptic Quality exactly as the organisers' self-evaluation notebook computes it.

Differences from the reference are in speed only: IoU is computed on compressed RLEs
with pycocotools instead of on dense float tensors, which gives the same values.

Reference rules, reproduced here:
- a GT/prediction pair is a true positive when IoU > 0.5 (strict);
- a prediction overlapping no GT above 0.5 is a false positive, and vice versa;
- an entry with no predictions adds all its GT as false negatives;
- PQ = sum(TP IoU) / (TP + 0.5 FP + 0.5 FN), pooled over all entries (not averaged).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_util

IOU_THRESHOLD = 0.5


def iou_matrix(gt: list[dict], pred: list[dict]) -> np.ndarray:
    """IoU between every GT RLE (rows) and every predicted RLE (columns)."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)))
    # pycocotools returns (n_dt, n_gt) for iou(dt, gt, iscrowd)
    return np.asarray(mask_util.iou(pred, gt, [0] * len(gt))).T.reshape(len(gt), len(pred))


@dataclass
class PQCounts:
    """Pooled PQ terms; add them across entries, then call ``score``."""

    tp_iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    multi: int = 0  # GT hit by more than one prediction: only overlapping predictions cause it

    def add(self, iou: np.ndarray, n_gt: int, n_pred: int) -> None:
        if n_gt == 0:
            self.fp += n_pred
            return
        if n_pred == 0:
            self.fn += n_gt
            return
        hit = iou > IOU_THRESHOLD
        self.tp_iou_sum += float(iou[hit].sum())
        self.tp += int(hit.sum())
        self.fp += int((hit.sum(axis=0) == 0).sum())
        self.fn += int((hit.sum(axis=1) == 0).sum())
        self.multi += int((hit.sum(axis=1) > 1).sum())

    def __iadd__(self, other: "PQCounts") -> "PQCounts":
        self.tp_iou_sum += other.tp_iou_sum
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.multi += other.multi
        return self

    def score(self) -> float:
        denominator = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp_iou_sum / denominator if denominator > 0 else 0.0

    def summary(self) -> str:
        sq = self.tp_iou_sum / self.tp if self.tp else 0.0
        text = f"PQ={self.score():.4f} SQ={sq:.4f} TP={self.tp} FP={self.fp} FN={self.fn}"
        return text + (f" DOUBLE-COUNTED={self.multi}" if self.multi else "")


def entry_counts(gt: list[dict], pred: list[dict]) -> PQCounts:
    """PQ terms for one annotator-image entry."""
    counts = PQCounts()
    counts.add(iou_matrix(gt, pred), len(gt), len(pred))
    return counts


def pooled_pq(gt_by_key: dict[str, list[dict]], pred_by_stem: dict[str, list[dict]]) -> PQCounts:
    """Score predictions (keyed by image stem) against every annotator entry.

    Args:
        gt_by_key: ``"<annotator>-<stem>"`` -> list of GT RLEs.
        pred_by_stem: ``stem`` -> list of predicted RLEs; missing stems count as empty.
    """
    total = PQCounts()
    for key, gt in gt_by_key.items():
        stem = key.split("-", 1)[1]
        total += entry_counts(gt, pred_by_stem.get(stem, []))
    return total
