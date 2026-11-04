"""Solar-disk geometry for GONG H-alpha images.

MAGFiLO annotators were asked to skip filaments more than about 70 degrees from the
central meridian (Reche & Cid 2026), so predictions close to the east or west limb are
mostly false positives. The disk centre and radius vary between images by tens of
pixels, so they are fitted per image rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from filseg.data import SIZE


@dataclass(frozen=True)
class Disk:
    cx: float  # pixels at SIZE resolution
    cy: float
    r: float


def fit_disk(image: np.ndarray, fraction: float = 0.3) -> Disk:
    """Fit the solar disk to the bright limb ring with a Hough circle transform.

    GONG images carry a scattered-light halo outside the limb, so a fit to all bright
    pixels overestimates the radius, by up to 14% on halo-heavy images. The limb itself
    is a sharp circular edge at 0.40-0.47 of the image width, which the Hough transform
    on a 512 px copy finds reliably. Falls back to the bright-pixel fit if no circle is
    found. Works on an image at any square resolution; the result is scaled to SIZE.
    """
    import cv2

    small = cv2.resize(np.asarray(image, np.uint8), (512, 512), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=60,
                               param2=20, minRadius=int(0.40 * 512), maxRadius=int(0.47 * 512))
    if circles is not None:
        cx, cy, r = (float(v) * SIZE / 512 for v in circles[0, 0])
        return Disk(cx=cx, cy=cy, r=r)
    return fit_disk_bright(image, fraction)


def fit_disk_bright(image: np.ndarray, fraction: float = 0.3) -> Disk:
    """Fit the disk from pixels brighter than ``fraction`` of the 99th percentile.

    Includes the halo outside the limb, so the radius runs large; kept as a fallback.
    """
    scale = SIZE / image.shape[0]
    bright = image.astype(np.float32) > fraction * np.percentile(image, 99)
    ys, xs = np.nonzero(bright)
    return Disk(cx=float(xs.mean()) * scale, cy=float(ys.mean()) * scale,
                r=float(np.sqrt(bright.sum() / np.pi)) * scale)


def longitude_deg(x: np.ndarray, y: np.ndarray, disk: Disk) -> np.ndarray:
    """Absolute heliographic longitude from the central meridian, ignoring the B0 tilt.

    Points at or beyond the limb return 90.
    """
    u = (np.asarray(x, np.float64) - disk.cx) / disk.r
    v = (disk.cy - np.asarray(y, np.float64)) / disk.r
    cos_lat = np.sqrt(np.clip(1.0 - v * v, 1e-9, None))
    return np.degrees(np.arcsin(np.clip(np.abs(u) / cos_lat, 0.0, 1.0)))
