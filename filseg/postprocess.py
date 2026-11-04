"""Turn a low-resolution filament probability map into scored instance RLEs.

Two stages, so tuning can reuse the expensive one:

1. ``components``: upsample to 2048, threshold, optionally close small gaps (used only
   to decide grouping), take 8-connected components, and describe each one by its
   area, mean probability (a confidence) and centroid longitude.
2. ``select``: keep components that are large enough, confident enough and not near
   the east or west limb, where MAGFiLO annotators did not label filaments.

Every false positive costs half a point of PQ, so dropping doubtful components pays.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from pycocotools import mask as mask_util

from filseg.data import SIZE
from filseg.geometry import Disk, longitude_deg


@dataclass(frozen=True)
class PostConfig:
    threshold: float = 0.5
    close_px: int = 0  # elliptical closing diameter at 2048 px; 0 disables it
    min_area: int = 200  # pixels at 2048 px
    min_score: float = 0.0  # mean probability inside the component
    max_lon: float = 90.0  # degrees from the central meridian; 90 disables the filter


@dataclass
class Component:
    rle: dict
    area: int
    score: float
    lon: float


def upsample(prob: np.ndarray) -> np.ndarray:
    """Bilinear resize of a (R, R) probability map to (SIZE, SIZE)."""
    if prob.shape[0] == SIZE:
        return prob.astype(np.float32)
    return cv2.resize(prob.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def components(prob: np.ndarray, threshold: float, close_px: int, disk: Disk | None,
               min_area: int = 1) -> list[Component]:
    """All connected components of the thresholded map, with their statistics."""
    full = upsample(prob)
    binary = (full > threshold).astype(np.uint8)
    grouping = binary
    if close_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        grouping = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n, labels = cv2.connectedComponents(grouping, connectivity=8)
    if n <= 1:
        return []
    # closing only decides grouping; the scored mask keeps the thresholded pixels
    labels[binary == 0] = 0
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=n)
    sums = np.bincount(flat, weights=full.ravel(), minlength=n)
    ys, xs = np.divmod(np.arange(flat.size), SIZE)
    cx = np.bincount(flat, weights=xs, minlength=n) / np.maximum(areas, 1)
    cy = np.bincount(flat, weights=ys, minlength=n) / np.maximum(areas, 1)
    lons = longitude_deg(cx, cy, disk) if disk is not None else np.zeros(n)
    out = []
    for label in range(1, n):
        if areas[label] < min_area:
            continue
        mask = np.asfortranarray((labels == label).astype(np.uint8))
        out.append(Component(mask_util.encode(mask), int(areas[label]),
                             float(sums[label] / areas[label]), float(lons[label])))
    return out


def keep(comps: list[Component], cfg: PostConfig) -> list[int]:
    """Indices of components that pass the area, confidence and limb filters."""
    return [i for i, c in enumerate(comps)
            if c.area >= cfg.min_area and c.score >= cfg.min_score and c.lon <= cfg.max_lon]


def instances(prob: np.ndarray, cfg: PostConfig = PostConfig(), disk: Disk | None = None) -> list[dict]:
    """Instance RLEs for one image from its probability map (any square resolution)."""
    comps = components(prob, cfg.threshold, cfg.close_px, disk, min_area=cfg.min_area)
    return [comps[i].rle for i in keep(comps, cfg)]
