"""Loading MAGFiLO annotations and images.

The training JSON is COCO-style. One H-alpha image may be annotated by up to three
annotators independently; the scorer treats each ``<annotator>-<image>`` entry as its
own ground truth, so this module keeps them apart rather than merging them.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ["FILSEG_RAW"]) if "FILSEG_RAW" in os.environ else ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMAGES = RAW / "train" / "train_images"
TEST_IMAGES = RAW / "test" / "test_images"
TRAIN_JSON = RAW / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

SIZE = 2048  # every GONG H-alpha image and every scored mask is SIZE x SIZE


@dataclass
class AnnotatorImage:
    """One annotator's full set of filaments for one image.

    Attributes:
        key: ``"<annotator>-<image_stem>"``, the unit the scorer evaluates.
        stem: Image file stem, e.g. ``"20140609195854Bh"``.
        rles: Compressed COCO RLE dicts, one per filament, at SIZE x SIZE.
    """

    key: str
    stem: str
    rles: list[dict] = field(default_factory=list)

    @property
    def annotator(self) -> str:
        return self.key.split("-", 1)[0]


def polygon_to_rle(polygon: list[float]) -> dict:
    """Rasterise one closed polygon ``[x0, y0, x1, y1, ...]`` to a compressed RLE."""
    rles = mask_util.frPyObjects([polygon], SIZE, SIZE)
    return mask_util.merge(rles)


def load_train(path: Path = TRAIN_JSON) -> list[AnnotatorImage]:
    """Return every annotator-image entry of the training set, sorted by key."""
    coco = json.loads(path.read_text())
    stem_of = {img["id"]: Path(img["file_name"]).stem for img in coco["images"]}
    entries = {key: AnnotatorImage(key=key, stem=stem) for key, stem in stem_of.items()}
    for ann in coco["annotations"]:
        for polygon in ann["segmentation"]:
            entries[ann["image_id"]].rles.append(polygon_to_rle(polygon))
    return sorted(entries.values(), key=lambda e: e.key)


def group_by_stem(entries: list[AnnotatorImage]) -> dict[str, list[AnnotatorImage]]:
    """Map each image stem to all of its annotator entries."""
    groups: dict[str, list[AnnotatorImage]] = defaultdict(list)
    for entry in entries:
        groups[entry.stem].append(entry)
    return dict(groups)


def read_image(path: Path) -> np.ndarray:
    """Read a grayscale H-alpha JPEG as a ``(SIZE, SIZE)`` uint8 array."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image
