"""Limb-darkening correction: divide each image by its own radial brightness profile.

H-alpha disks darken towards the limb, so a dark filament near the limb and the quiet
Sun near the centre can have the same grey level. Dividing by the median intensity of
each thin ring around the fitted disk centre makes the quiet Sun flat (about 128 after
rescaling); filaments stay darker than their ring and plage brighter. Pixels off the
disk are set to 0. After Reche & Cid 2026, who report that this correction helps.

Run: ``.venv/bin/python -m filseg.flatten`` writes ``train_images_flat.npy`` and
``test_images_flat.npy`` next to the plain cache, from the plain cached images.

Status: not used by the submitted pipeline (submission 8). The limb-darkening-corrected input (``--images flat``) gave
no gain on fold 0 (tuned PQ 0.4112 against 0.4160 for the plain input). docs/DESIGN.md, "Limb-darkening
correction as input, fold 0".
"""

from __future__ import annotations

import numpy as np

from filseg.cache import CACHE
from filseg.geometry import fit_disk

RING_PX = 4  # ring width at cache resolution
QUIET = 128.0  # grey level the quiet Sun is mapped to


def flatten(image: np.ndarray) -> np.ndarray:
    """Limb-darkening-corrected copy of one cached ``(R, R)`` uint8 image."""
    res = image.shape[0]
    disk = fit_disk(image)
    scale = res / 2048
    cx, cy, r = disk.cx * scale, disk.cy * scale, disk.r * scale
    ys, xs = np.mgrid[0:res, 0:res]
    rad = np.hypot(xs - cx, ys - cy)
    on = rad < r * 0.995
    ring = (rad / RING_PX).astype(int)
    values = image.astype(np.float32)
    n_rings = int(ring[on].max()) + 1
    profile = np.zeros(n_rings, np.float32)
    for k in range(n_rings):  # median per ring is robust to filaments and plage
        sel = values[on & (ring == k)]
        profile[k] = np.median(sel) if sel.size else 0.0
    # rings with few pixels at the centre are noisy: smooth the profile a little
    kernel = np.ones(5, np.float32) / 5
    smooth = np.convolve(np.pad(profile, 2, mode="edge"), kernel, mode="valid")
    smooth = np.maximum(smooth, 1.0)
    out = np.zeros_like(values)
    out[on] = values[on] / smooth[ring[on]] * QUIET
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    from tqdm import tqdm

    for split in ("train", "test"):
        src = np.load(CACHE / f"{split}_images.npy", mmap_mode="r")
        dst = np.lib.format.open_memmap(CACHE / f"{split}_images_flat.npy", "w+", np.uint8, src.shape)
        for i in tqdm(range(len(src)), desc=split):
            dst[i] = flatten(np.array(src[i]))
        dst.flush()


if __name__ == "__main__":
    main()
