"""Sanity checks and score ceilings computed from the training annotations alone.

1. The RLE-based PQ matches the organisers' dense-tensor implementation.
2. Inter-annotator PQ: one annotator's masks scored as predictions against another's.
3. Connected-components ceiling: an annotator's own union mask, split into connected
   components, scored against that annotator. This is the best a semantic model with
   plain connected-component splitting could reach.
"""

from __future__ import annotations

import itertools
import random

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util

from filseg.data import load_train, group_by_stem
from filseg.metric import PQCounts, entry_counts


def dense_reference_pq(pairs: list[tuple[list[dict], list[dict]]]) -> float:
    """PQ via the organisers' dense matmul code path, for cross-checking."""
    tp_scores, fp, fn = [], 0, 0
    for gt, pred in pairs:
        g = torch.from_numpy(mask_util.decode(gt).transpose(2, 0, 1).astype(np.float32))
        p = torch.from_numpy(mask_util.decode(pred).transpose(2, 0, 1).astype(np.float32))
        gf, pf = g.reshape(len(gt), -1), p.reshape(len(pred), -1)
        inter = gf @ pf.t()
        union = gf.sum(1).view(-1, 1) + pf.sum(1).view(1, -1) - inter
        iou = torch.where(union == 0, torch.tensor(0.0), inter / union)
        hit = iou > 0.5
        tp_scores += iou[hit].tolist()
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    return sum(tp_scores) / (len(tp_scores) + 0.5 * fp + 0.5 * fn)


def components(rles: list[dict], min_area: int = 0) -> list[dict]:
    """Union the RLEs, then split the union into 8-connected components."""
    union = mask_util.decode(mask_util.merge(rles))
    n, labels = cv2.connectedComponents(union, connectivity=8)
    out = []
    for label in range(1, n):
        component = labels == label
        if component.sum() >= min_area:
            out.append(mask_util.encode(np.asfortranarray(component.astype(np.uint8))))
    return out


def main() -> None:
    entries = load_train()
    groups = group_by_stem(entries)
    print(f"{len(entries)} annotator-images over {len(groups)} images")

    multi = [g for g in groups.values() if len(g) > 1]
    rng = random.Random(0)

    pairs = []
    for g in rng.sample(multi, 15):
        a, b = g[0], g[1]
        pairs.append((a.rles, b.rles))
    fast = PQCounts()
    for gt, pred in pairs:
        fast += entry_counts(gt, pred)
    print(f"metric check: rle={fast.score():.6f} dense={dense_reference_pq(pairs):.6f}")

    inter = PQCounts()
    for g in multi:
        for a, b in itertools.permutations(g, 2):
            inter += entry_counts(a.rles, b.rles)
    print(f"inter-annotator ({len(multi)} images, all ordered pairs): {inter.summary()}")

    cc = PQCounts()
    for e in entries:
        cc += entry_counts(e.rles, components(e.rles))
    print(f"own-union connected components (all entries): {cc.summary()}")


if __name__ == "__main__":
    main()
