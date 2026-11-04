"""Predict the test set and write a validated Kaggle submission CSV.

Each row is one predicted filament: ``filament_id`` = ``<image stem>_<n>`` and
``segmentation_rle`` = compressed COCO RLE counts of a 2048 x 2048 mask. Images with no
predicted filament get no rows; the scorer counts their annotations as misses.

Run: ``.venv/bin/python -m filseg.predict --weights runs/unet_r34_f-1/last.pt --post runs/unet_r34_f0/tune.json``
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

from filseg.cache import CACHE
from filseg.data import ROOT, SIZE, TEST_IMAGES, read_image
from filseg.geometry import fit_disk
from filseg.postprocess import PostConfig, instances, upsample
from filseg.scorer import load_scorer
from filseg.train import build_model, image_cache, predict

SUBMISSIONS = ROOT / "submissions"


def load_post(path: Path | None) -> PostConfig:
    """Best setting from a ``tune.json``, or the defaults."""
    if path is None:
        return PostConfig()
    best = json.loads(path.read_text())[0]
    return PostConfig(best["threshold"], best["close_px"], best["min_area"], best["min_score"], best["max_lon"])


def write_csv(rows: list[tuple[str, str]], path: Path, stems: list[str]) -> None:
    """Write the submission and check every row decodes to a non-empty 2048 mask."""
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate filament_id"
    known = set(stems)
    for filament_id, counts in rows:
        assert filament_id.rsplit("_", 1)[0] in known, filament_id
        area = mask_util.area({"size": [SIZE, SIZE], "counts": counts.encode()})
        assert area > 0, f"empty mask in {filament_id}"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", required=True, help="one or more checkpoints to average")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--post", type=Path, default=None, help="tune.json to take the best setting from")
    parser.add_argument("--name", default="submission")
    parser.add_argument("--images", default="plain", choices=["plain", "flat"])
    parser.add_argument("--views", type=int, default=2, choices=[1, 2, 4], help="flip views to average")
    parser.add_argument("--scorer", type=Path, default=None,
                        help="component scorer pickle (filseg.scorer); replaces the --post rule")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    stems = json.loads((CACHE / "index.json").read_text())["test"]
    images = np.load(CACHE / image_cache("test", args.images), mmap_mode="r")
    cfg = load_post(args.post)
    scorer = load_scorer(args.scorer) if args.scorer else None
    started = time.time()

    probs = np.zeros(images.shape, np.float32)
    for weights in args.weights:
        model = build_model(args.encoder, pretrained=False).to(device)
        model.load_state_dict(torch.load(weights, map_location=device))
        probs += predict(model, images, device, views=args.views).astype(np.float32) / len(args.weights)

    rows = []
    plain = np.load(CACHE / "test_images.npy", mmap_mode="r")  # the disk is fitted on the plain image
    for k, stem in enumerate(stems):
        disk = fit_disk(np.array(plain[k]))
        if scorer is not None:
            kept = scorer.select(upsample(probs[k]), read_image(TEST_IMAGES / f"{stem}.jpeg"), disk)
        else:
            kept = instances(probs[k], cfg, disk)
        for n, rle in enumerate(kept, start=1):
            rows.append((f"{stem}_{n}", rle["counts"].decode()))
    elapsed = time.time() - started

    SUBMISSIONS.mkdir(exist_ok=True)
    out = SUBMISSIONS / f"{args.name}.csv"
    write_csv(rows, out, stems)
    with_preds = len({r[0].rsplit('_', 1)[0] for r in rows})
    selection = {"scorer": str(args.scorer), "tau": scorer.tau} if scorer else cfg.__dict__
    meta = {"weights": args.weights, "views": args.views, "post": selection, "rows": len(rows),
            "images_with_predictions": with_preds, "images": len(stems),
            "seconds": round(elapsed, 1), "seconds_per_image": round(elapsed / len(stems), 3)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
