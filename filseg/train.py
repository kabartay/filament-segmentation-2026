"""Train a U-Net on the cached consensus targets and validate with the exact PQ.

Folds group images by calendar month: filaments persist for days, so images from the
same week would otherwise leak between train and validation.

Run: ``.venv/bin/python -m filseg.train --fold 0 --name unet_r34``
Use ``--fold -1`` to train on all images with no validation (for the submission model).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

from filseg.cache import CACHE
from filseg.data import ROOT, group_by_stem, load_train
from filseg.metric import PQCounts, entry_counts
from filseg.postprocess import PostConfig, instances

RUNS = ROOT / "runs"
N_FOLDS = 5


def fold_split(stems: list[str], fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of (train, validation) stems for ``fold``, grouped by YYYYMM."""
    months = [stem[:6] for stem in stems]
    splits = list(GroupKFold(n_splits=N_FOLDS).split(stems, groups=months))
    return splits[fold]


def image_cache(split: str, kind: str = "plain") -> str:
    """File name of the cached images: plain, or limb-darkening-corrected (``filseg.flatten``)."""
    return f"{split}_images.npy" if kind == "plain" else f"{split}_images_flat.npy"


def normalise(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) / 255.0 - 0.5) / 0.25


class CachedSet(Dataset):
    """Images and soft targets from the memmapped cache.

    The memmaps are opened lazily so each loader worker maps the file itself instead of
    receiving a pickled copy of the arrays.
    """

    def __init__(self, indices: np.ndarray, augment: bool, images_file: str = "train_images.npy"):
        self.indices = indices
        self.images_file = images_file
        self.images = self.targets = None
        self.transform = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Affine(rotate=(-30, 30), scale=(0.9, 1.1), border_mode=0, p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
            A.RandomGamma((80, 120), p=0.3),
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self.images is None:
            self.images = np.load(CACHE / self.images_file, mmap_mode="r")
            self.targets = np.load(CACHE / "train_targets.npy", mmap_mode="r")
        k = self.indices[i]
        image, target = np.array(self.images[k]), np.array(self.targets[k])
        if self.transform is not None:
            out = self.transform(image=image, mask=target)
            image, target = out["image"], out["mask"]
        x = torch.from_numpy(normalise(image))[None]
        y = torch.from_numpy(target.astype(np.float32) / 255.0)[None]
        return x, y


def build_model(encoder: str, pretrained: bool = True) -> torch.nn.Module:
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=1, classes=1)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + 1) / (denom + 1)).mean()


@torch.no_grad()
def predict(model: torch.nn.Module, images: np.ndarray, device: str, tta: bool = True) -> np.ndarray:
    """Sigmoid probabilities (N, R, R) float16, averaged over a horizontal flip."""
    model.eval()
    out = np.empty(images.shape, dtype=np.float16)
    for k in range(len(images)):
        x = torch.from_numpy(normalise(np.array(images[k])))[None, None].to(device)
        p = torch.sigmoid(model(x))
        if tta:
            p = (p + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
        out[k] = p[0, 0].float().cpu().numpy()
    return out


def score_probs(probs: np.ndarray, stems: list[str], groups: dict, cfg: PostConfig) -> PQCounts:
    total = PQCounts()
    for prob, stem in zip(probs, stems):
        pred = instances(prob, cfg)
        for entry in groups[stem]:
            total += entry_counts(entry.rles, pred)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--name", default="unet_r34")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--loss", default="bce_dice", choices=["bce_dice", "bce_lovasz"],
                        help="term added to BCE: soft Dice, or Lovasz hinge on the majority mask")
    parser.add_argument("--images", default="plain", choices=["plain", "flat"],
                        help="flat: limb-darkening-corrected cache from filseg.flatten")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_dir = RUNS / f"{args.name}_f{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=1))
    log = (run_dir / "log.txt").open("a")

    def emit(message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    index = json.loads((CACHE / "index.json").read_text())
    stems = index["train"]
    if args.fold >= 0:
        train_idx, val_idx = fold_split(stems, args.fold)
    else:
        train_idx, val_idx = np.arange(len(stems)), np.array([], dtype=int)
    groups = group_by_stem(load_train()) if len(val_idx) else {}
    val_stems = [stems[k] for k in val_idx]
    images_file = image_cache("train", args.images)
    val_images = np.load(CACHE / images_file, mmap_mode="r")[val_idx] if len(val_idx) else None
    emit(f"device={device} train={len(train_idx)} val={len(val_idx)} args={vars(args)}")

    loader = DataLoader(CachedSet(train_idx, augment=True, images_file=images_file), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True)
    model = build_model(args.encoder).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(loader)
    warmup = len(loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    if args.loss == "bce_lovasz":
        # Lovasz hinge optimises IoU directly; it needs a binary target, so use the majority mask
        lovasz = smp.losses.LovaszLoss("binary", per_image=True)
        second_loss = lambda logits, y: lovasz(logits, (y >= 0.5).float())  # noqa: E731
    else:
        second_loss = soft_dice_loss
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + second_loss(logits, y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            running += loss.item()
        emit(f"epoch {epoch} loss={running / len(loader):.4f} {time.time() - started:.0f}s")

        if val_images is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            probs = predict(model, val_images, device)
            counts = score_probs(probs, val_stems, groups, PostConfig())
            emit(f"  val {counts.summary()}")
            if counts.score() > best:
                best = counts.score()
                torch.save(model.state_dict(), run_dir / "best.pt")
                np.save(run_dir / "val_probs.npy", probs)
                (run_dir / "val_stems.json").write_text(json.dumps(val_stems))
    torch.save(model.state_dict(), run_dir / "last.pt")
    emit(f"done best_val_pq={best:.4f}")


if __name__ == "__main__":
    main()
