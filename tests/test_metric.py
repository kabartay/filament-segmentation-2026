"""Panoptic Quality (``filseg.metric``) on toy masks: strict IoU > 0.5, FP/FN counts and pooling."""

from __future__ import annotations

import numpy as np
import pytest
from pycocotools import mask as mask_util

from filseg.metric import PQCounts, entry_counts, iou_matrix, pooled_pq

H = W = 64


def rle(rows: slice, cols: slice) -> dict:
    """RLE of a rectangle in a 64 x 64 image."""
    m = np.zeros((H, W), np.uint8)
    m[rows, cols] = 1
    return mask_util.encode(np.asfortranarray(m))


def test_identical_prediction_scores_one() -> None:
    gt = [rle(slice(0, 10), slice(0, 10)), rle(slice(30, 40), slice(30, 50))]
    counts = entry_counts(gt, list(gt))
    assert (counts.tp, counts.fp, counts.fn) == (2, 0, 0)
    assert counts.score() == pytest.approx(1.0)


def test_empty_prediction_scores_zero_and_counts_fn() -> None:
    gt = [rle(slice(0, 10), slice(0, 10)), rle(slice(30, 40), slice(30, 50))]
    counts = entry_counts(gt, [])
    assert (counts.tp, counts.fp, counts.fn) == (0, 0, 2)
    assert counts.score() == 0.0


def test_empty_gt_counts_all_predictions_as_fp() -> None:
    counts = entry_counts([], [rle(slice(0, 5), slice(0, 5))])
    assert (counts.tp, counts.fp, counts.fn) == (0, 1, 0)
    assert counts.score() == 0.0


def test_nothing_at_all_is_zero_not_a_division_error() -> None:
    assert entry_counts([], []).score() == 0.0


def test_iou_exactly_half_is_not_a_match() -> None:
    gt = [rle(slice(0, 10), slice(0, 2))]  # 20 px
    pred = [rle(slice(0, 10), slice(0, 1))]  # 10 px inside it: IoU = 10 / 20 = 0.5 exactly
    assert iou_matrix(gt, pred)[0, 0] == 0.5
    counts = entry_counts(gt, pred)
    assert (counts.tp, counts.fp, counts.fn) == (0, 1, 1)
    assert counts.score() == 0.0


def test_iou_just_above_half_is_a_match() -> None:
    gt = [rle(slice(0, 10), slice(0, 10))]  # 100 px
    pred = [rle(slice(0, 10), slice(0, 6))]  # 60 px inside it: IoU = 0.6
    counts = entry_counts(gt, pred)
    assert (counts.tp, counts.fp, counts.fn) == (1, 0, 0)
    assert counts.score() == pytest.approx(0.6)


def test_fp_and_fn_each_cost_half_a_point() -> None:
    gt = [rle(slice(0, 10), slice(0, 10)), rle(slice(20, 30), slice(0, 10))]
    pred = [rle(slice(0, 10), slice(0, 10)), rle(slice(50, 60), slice(50, 60))]
    counts = entry_counts(gt, pred)
    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 1)
    assert counts.score() == pytest.approx(1.0 / (1 + 0.5 + 0.5))


def test_overlapping_predictions_flag_double_counting() -> None:
    gt = [rle(slice(0, 10), slice(0, 10))]
    pred = [rle(slice(0, 10), slice(0, 8)), rle(slice(0, 10), slice(2, 10))]
    counts = entry_counts(gt, pred)
    assert counts.tp == 2 and counts.multi == 1
    assert "DOUBLE-COUNTED=1" in counts.summary()


def test_iou_matrix_shape_is_gt_by_pred() -> None:
    gt = [rle(slice(0, 4), slice(0, 4))] * 3
    pred = [rle(slice(0, 4), slice(0, 4))] * 2
    assert iou_matrix(gt, pred).shape == (3, 2)
    assert iou_matrix([], pred).shape == (0, 2)
    assert iou_matrix(gt, []).shape == (3, 0)


def test_pooled_pq_is_pooled_not_averaged() -> None:
    a = rle(slice(0, 10), slice(0, 10))
    b = rle(slice(20, 30), slice(20, 30))
    c = rle(slice(40, 50), slice(40, 50))
    gt = {
        "ann1-img1": [a],              # perfect: 1 TP
        "ann2-img1": [a],              # same image, second annotator: another TP
        "ann1-img2": [b, c],           # image 2 has no prediction: 2 FN
    }
    pred = {"img1": [a]}               # img2 stem missing -> counted as empty
    total = pooled_pq(gt, pred)
    assert (total.tp, total.fp, total.fn) == (2, 0, 2)
    # pooled: 2 / (2 + 0.5 * 2) = 2/3; an average of per-entry scores (1, 1, 0) would also be 2/3,
    # so use an asymmetric case below to tell them apart
    assert total.score() == pytest.approx(2 / 3)


def test_pooled_pq_differs_from_mean_of_entries() -> None:
    a = rle(slice(0, 10), slice(0, 10))
    b = rle(slice(20, 30), slice(20, 30))
    c = rle(slice(40, 50), slice(40, 50))
    d = rle(slice(50, 60), slice(0, 10))
    gt = {"x-big": [a, b, c, d], "y-small": [a]}
    pred = {"big": [a, b, c, d], "small": []}
    total = pooled_pq(gt, pred)
    assert (total.tp, total.fn) == (4, 1)
    assert total.score() == pytest.approx(4 / (4 + 0.5))
    mean_of_entries = (1.0 + 0.0) / 2
    assert total.score() != pytest.approx(mean_of_entries)


def test_key_stem_split_only_on_first_dash() -> None:
    a = rle(slice(0, 10), slice(0, 10))
    total = pooled_pq({"ann-img-with-dashes": [a]}, {"img-with-dashes": [a]})
    assert total.tp == 1 and total.score() == pytest.approx(1.0)


def test_iadd_accumulates_every_field() -> None:
    x = PQCounts(1.5, 2, 3, 4, 1)
    x += PQCounts(0.5, 1, 1, 1, 0)
    assert (x.tp_iou_sum, x.tp, x.fp, x.fn, x.multi) == (2.0, 3, 4, 5, 1)
