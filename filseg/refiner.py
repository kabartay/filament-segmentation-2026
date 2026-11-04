"""Per-filament crop refiner at native 2048 px resolution.

A first-stage model (the U-Net or the detector) decides which filaments exist; this
model redraws each one's mask from a full-resolution crop, where thin barbs survive
that are lost at 1024 px. Following Reche & Cid (2026), each crop covers about three
times the filament's bounding box, and a second input channel marks the target box so
the network segments that filament and not a neighbour sharing the crop.

Training samples are the annotated filaments of every annotator, with the box jittered
so the model tolerates imperfect first-stage boxes.

Run:
  ``.venv/bin/python -m filseg.refiner cache``                      build the crop store
  ``.venv/bin/python -m filseg.refiner train --fold 0``             train
  ``.venv/bin/python -m filseg.refiner eval --fold 0 --proposals sub1``  PQ on U-Net proposals
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader, Dataset

from filseg.cache import CACHE
from filseg.data import SIZE, TRAIN_IMAGES, group_by_stem, load_train, read_image
from filseg.metric import PQCounts, entry_counts
from filseg.train import RUNS, fold_split, normalise, soft_dice_loss

STORE = 320  # stored crop side in pixels
STORE_CONTEXT = 4.0  # stored crops cover 4x the box, leaving room for jitter
CONTEXT = 3.0  # the model sees about 3x the box
INPUT = 256  # model input side
MIN_SIDE = 96  # smallest crop window at 2048 px


def window(box: tuple[float, float, float, float], context: float) -> tuple[float, float, float]:
    """Square window (x0, y0, side) centred on an (x, y, w, h) box."""
    x, y, w, h = box
    side = max(context * max(w, h), MIN_SIDE)
    return x + w / 2 - side / 2, y + h / 2 - side / 2, side


def crop(image: np.ndarray, x0: float, y0: float, side: float, out: int, nearest: bool = False) -> np.ndarray:
    """Resample the square window of ``image`` to ``out`` x ``out``, zero outside the image."""
    ix0, iy0, iside = int(round(x0)), int(round(y0)), max(int(round(side)), 1)
    canvas = np.zeros((iside, iside), dtype=image.dtype)
    sx0, sy0 = max(ix0, 0), max(iy0, 0)
    sx1, sy1 = min(ix0 + iside, image.shape[1]), min(iy0 + iside, image.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = image[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if iside > out else cv2.INTER_LINEAR)
    return cv2.resize(canvas, (out, out), interpolation=interp)


def box_channel(box: tuple[float, float, float, float], x0: float, y0: float, side: float, out: int) -> np.ndarray:
    """1 inside the target box, 0 elsewhere, in the coordinates of a crop window."""
    s = out / side
    x, y, w, h = box
    channel = np.zeros((out, out), np.float32)
    c0, r0 = int(max((x - x0) * s, 0)), int(max((y - y0) * s, 0))
    c1, r1 = int(min(math.ceil((x + w - x0) * s), out)), int(min(math.ceil((y + h - y0) * s), out))
    channel[r0:r1, c0:c1] = 1.0
    return channel


def build_store() -> None:
    """Cache a 4x-context crop of every annotated filament (all annotators)."""
    groups = group_by_stem(load_train())
    rows, n = [], sum(len(e.rles) for g in groups.values() for e in g)
    images = np.lib.format.open_memmap(CACHE / "crop_images.npy", "w+", np.uint8, (n, STORE, STORE))
    masks = np.lib.format.open_memmap(CACHE / "crop_masks.npy", "w+", np.uint8, (n, STORE, STORE))
    k = 0
    for stem in sorted(groups):
        image = read_image(TRAIN_IMAGES / f"{stem}.jpeg")
        for entry in groups[stem]:
            for rle in entry.rles:
                box = tuple(float(v) for v in mask_util.toBbox(rle))
                x0, y0, side = window(box, STORE_CONTEXT)
                images[k] = crop(image, x0, y0, side, STORE)
                full = mask_util.decode(rle)
                masks[k] = (crop(full * 255, x0, y0, side, STORE) > 127).astype(np.uint8)
                rows.append({"stem": stem, "box": box, "window": (x0, y0, side)})
                k += 1
    images.flush()
    masks.flush()
    (CACHE / "crop_index.json").write_text(json.dumps(rows))
    print(f"cached {k} filament crops at {STORE}px")


class CropSet(Dataset):
    """Training crops resampled from the store with jittered context and box."""

    def __init__(self, indices: list[int], rows: list[dict], train: bool):
        self.indices, self.rows, self.train = indices, rows, train
        self.images = self.masks = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self.images is None:
            self.images = np.load(CACHE / "crop_images.npy", mmap_mode="r")
            self.masks = np.load(CACHE / "crop_masks.npy", mmap_mode="r")
        k = self.indices[i]
        row = self.rows[k]
        wx, wy, wside = row["window"]
        s = STORE / wside  # store pixels per 2048 pixel
        x, y, w, h = row["box"]
        box = ((x - wx) * s, (y - wy) * s, w * s, h * s)  # in store coordinates
        context = CONTEXT
        if self.train:
            context = random.uniform(2.5, 3.5)
            size = max(box[2], box[3])
            grow = random.uniform(0.9, 1.15)
            bw, bh = box[2] * grow, box[3] * grow
            bx = box[0] + box[2] / 2 - bw / 2 + random.uniform(-0.05, 0.05) * size
            by = box[1] + box[3] / 2 - bh / 2 + random.uniform(-0.05, 0.05) * size
            box = (bx, by, bw, bh)
        x0, y0, side = window(box, context)
        side = max(side, MIN_SIDE * s)
        image = crop(np.array(self.images[k]), x0, y0, side, INPUT)
        target = crop(np.array(self.masks[k]), x0, y0, side, INPUT, nearest=True).astype(np.float32)
        channel = box_channel(box, x0, y0, side, INPUT)
        if self.train:
            turns = random.randrange(4)
            image, target, channel = (np.rot90(a, turns) for a in (image, target, channel))
            if random.random() < 0.5:
                image, target, channel = (a[:, ::-1] for a in (image, target, channel))
            image = np.clip(255.0 * (image / 255.0) ** random.uniform(0.75, 1.35) * random.uniform(0.9, 1.1), 0, 255)
        x = np.stack([normalise(np.ascontiguousarray(image)), np.ascontiguousarray(channel)])
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(np.ascontiguousarray(target))[None]


def build_refiner(pretrained: bool = True) -> torch.nn.Module:
    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=2, classes=1)


@torch.no_grad()
def refine(model: torch.nn.Module, image: np.ndarray, rles: list[dict], device: str,
           threshold: float = 0.5, batch: int = 32) -> list[dict]:
    """Redraw each instance from a native-resolution crop.

    Pixels claimed by several refined instances go to the one with the highest
    probability; an instance whose refined mask comes out empty keeps its original mask.
    """
    if not rles:
        return []
    model.eval()
    tasks = []
    for rle in rles:
        box = tuple(float(v) for v in mask_util.toBbox(rle))
        x0, y0, side = window(box, CONTEXT)
        x = np.stack([normalise(crop(image, x0, y0, side, INPUT)), box_channel(box, x0, y0, side, INPUT)])
        tasks.append((x0, y0, side, x))
    probs = []
    for start in range(0, len(tasks), batch):
        x = torch.from_numpy(np.stack([t[3] for t in tasks[start:start + batch]]).astype(np.float32)).to(device)
        p = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
        probs.extend(p[:, 0].cpu().numpy())

    best = np.zeros((SIZE, SIZE), np.float32)
    owner = np.full((SIZE, SIZE), -1, np.int32)
    for idx, ((x0, y0, side, _), prob) in enumerate(zip(tasks, probs)):
        iside = max(int(round(side)), 1)
        up = cv2.resize(prob, (iside, iside), interpolation=cv2.INTER_LINEAR)
        ix0, iy0 = int(round(x0)), int(round(y0))
        sx0, sy0 = max(ix0, 0), max(iy0, 0)
        sx1, sy1 = min(ix0 + iside, SIZE), min(iy0 + iside, SIZE)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        patch = up[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0]
        region_best = best[sy0:sy1, sx0:sx1]
        win = (patch > threshold) & (patch > region_best)
        region_best[win] = patch[win]
        owner[sy0:sy1, sx0:sx1][win] = idx
    out = []
    for idx, rle in enumerate(rles):
        mask = owner == idx
        out.append(mask_util.encode(np.asfortranarray(mask.astype(np.uint8))) if mask.any() else rle)
    return out


def train(args: argparse.Namespace) -> None:
    rows = json.loads((CACHE / "crop_index.json").read_text())
    stems = json.loads((CACHE / "index.json").read_text())["train"]
    train_idx, val_idx = fold_split(stems, args.fold)
    train_stems = {stems[k] for k in train_idx}
    tr = [i for i, r in enumerate(rows) if r["stem"] in train_stems]
    va = [i for i, r in enumerate(rows) if r["stem"] not in train_stems]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_dir = RUNS / f"refiner_f{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "log.txt").open("a")

    def emit(message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    random.seed(0)
    torch.manual_seed(0)
    emit(f"device={device} train_crops={len(tr)} val_crops={len(va)} args={vars(args)}")
    loader = DataLoader(CropSet(tr, rows, True), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(CropSet(va, rows, False), batch_size=args.batch, num_workers=args.workers)
    model = build_refiner().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: min(1.0, (s + 1) / len(loader)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + soft_dice_loss(logits, y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            running += loss.item()
        model.eval()
        inter = denom = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                p = (torch.sigmoid(model(x.to(device))) > 0.5).float().cpu()
                inter += float((p * y).sum())
                denom += float(p.sum() + y.sum())
        dice = 2 * inter / max(denom, 1.0)
        emit(f"epoch {epoch} loss={running / len(loader):.4f} val_dice={dice:.4f} {time.time() - started:.0f}s")
        if dice > best:
            best = dice
            torch.save(model.state_dict(), run_dir / "best.pt")
    emit(f"done best_val_dice={best:.4f}")


def evaluate(args: argparse.Namespace) -> None:
    """PQ of the first-stage proposals before and after refinement, on validation images."""
    from filseg.geometry import fit_disk
    from filseg.postprocess import PostConfig, instances
    from filseg.predict import load_post

    run_dir = RUNS / args.proposals
    stems = json.loads((run_dir / "val_stems.json").read_text())
    probs = np.load(run_dir / "val_probs.npy", mmap_mode="r")
    cfg = load_post(run_dir / "post.json") if (run_dir / "post.json").exists() else PostConfig()
    small = np.load(CACHE / "train_images.npy", mmap_mode="r")
    index = json.loads((CACHE / "index.json").read_text())["train"]
    groups = group_by_stem(load_train())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_refiner(pretrained=False).to(device)
    model.load_state_dict(torch.load(RUNS / f"refiner_f{args.fold}" / "best.pt", map_location=device))
    before = PQCounts()
    after = {t: PQCounts() for t in args.thresholds}
    for prob, stem in zip(probs, stems):
        disk = fit_disk(np.array(small[index.index(stem)]))
        proposals = instances(np.asarray(prob, np.float32), cfg, disk)
        image = read_image(TRAIN_IMAGES / f"{stem}.jpeg")
        refined = {t: refine(model, image, proposals, device, threshold=t) for t in args.thresholds}
        for entry in groups[stem]:
            before += entry_counts(entry.rles, proposals)
            for t in args.thresholds:
                after[t] += entry_counts(entry.rles, refined[t])
    print(f"proposals ({cfg}): {before.summary()}")
    for t in args.thresholds:
        print(f"refined thr={t}: {after[t].summary()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache")
    t = sub.add_parser("train")
    t.add_argument("--fold", type=int, default=0)
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--workers", type=int, default=2)
    e = sub.add_parser("eval")
    e.add_argument("--fold", type=int, default=0)
    e.add_argument("--proposals", default="sub1")
    e.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    args = parser.parse_args()
    {"cache": lambda a: build_store(), "train": train, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
