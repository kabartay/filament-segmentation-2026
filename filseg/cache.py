"""Build the downsampled training cache used by ``filseg.train``.

Writes to ``data/cache/``:
- ``train_images.npy``: (N, R, R) uint8 grayscale, area-downsampled from 2048;
- ``train_targets.npy``: (N, R, R) uint8 soft target, 255 * fraction of annotators
  marking the pixel, computed at 2048 then area-downsampled;
- ``test_images.npy``: (M, R, R) uint8;
- ``index.json``: stems in array order, and the resolution R.

Run: ``.venv/bin/python -m filseg.cache --res 1024``
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util
from tqdm import tqdm

from filseg.data import ROOT, TEST_IMAGES, TRAIN_IMAGES, group_by_stem, load_train, read_image

# FILSEG_CACHE selects another cache, e.g. data/cache1536 for a 1536 px experiment
CACHE = Path(os.environ["FILSEG_CACHE"]).resolve() if "FILSEG_CACHE" in os.environ else ROOT / "data" / "cache"


def downsample(array: np.ndarray, res: int) -> np.ndarray:
    return cv2.resize(array, (res, res), interpolation=cv2.INTER_AREA)


def consensus(rle_sets: list[list[dict]]) -> np.ndarray:
    """Fraction of annotators marking each pixel, as float32 at full resolution."""
    total = None
    for rles in rle_sets:
        union = mask_util.decode(mask_util.merge(rles)).astype(np.float32)
        total = union if total is None else total + union
    return total / len(rle_sets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=1024)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    groups = group_by_stem(load_train())
    train_stems = sorted(groups)
    test_stems = sorted(p.stem for p in TEST_IMAGES.glob("*.jpeg"))

    images = np.lib.format.open_memmap(
        CACHE / "train_images.npy", "w+", np.uint8, (len(train_stems), args.res, args.res))
    targets = np.lib.format.open_memmap(
        CACHE / "train_targets.npy", "w+", np.uint8, (len(train_stems), args.res, args.res))
    for i, stem in enumerate(tqdm(train_stems, desc="train")):
        images[i] = downsample(read_image(TRAIN_IMAGES / f"{stem}.jpeg"), args.res)
        soft = consensus([entry.rles for entry in groups[stem]])
        targets[i] = np.round(downsample(soft, args.res) * 255).astype(np.uint8)
    images.flush()
    targets.flush()

    test = np.lib.format.open_memmap(
        CACHE / "test_images.npy", "w+", np.uint8, (len(test_stems), args.res, args.res))
    for i, stem in enumerate(tqdm(test_stems, desc="test")):
        test[i] = downsample(read_image(TEST_IMAGES / f"{stem}.jpeg"), args.res)
    test.flush()

    index = {"res": args.res, "train": train_stems, "test": test_stems}
    (CACHE / "index.json").write_text(json.dumps(index, indent=1))
    print(f"cached {len(train_stems)} train and {len(test_stems)} test at {args.res}px")


if __name__ == "__main__":
    main()
