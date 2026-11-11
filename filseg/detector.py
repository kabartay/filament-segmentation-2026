"""Mask R-CNN filament detector trained on the 1024 px cache.

Unlike the semantic U-Net, the detector predicts one mask per filament, so grouping
pixels into filaments is learned from the annotations instead of left to connected
components. Each epoch samples one annotator per image, so images with several
annotators contribute every annotator's grouping over the course of training.

Run: ``.venv/bin/python -m filseg.detector --fold 0 --name mrcnn``

Status: not used by the submitted pipeline (submission 8). Mask R-CNN on its own scored below the U-Net (fold 0, best
0.377 with disjoint masks against 0.389), and detector instances added to U-Net components gained at most 0.01,
inside fold noise. docs/DESIGN.md, "Mask R-CNN detector, fold 0".
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from filseg.cache import CACHE
from filseg.data import SIZE, group_by_stem, load_train
from filseg.metric import PQCounts, entry_counts
from filseg.train import RUNS, fold_split

RES = 1024


def instance_cache(res: int = RES) -> dict[str, list[list[dict]]]:
    """stem -> one list of RLEs (at ``res``) per annotator, built once and pickled."""
    path = CACHE / f"instances_{res}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    out: dict[str, list[list[dict]]] = {}
    for stem, entries in group_by_stem(load_train()).items():
        sets = []
        for entry in entries:
            rles = []
            for rle in entry.rles:
                small = cv2.resize(mask_util.decode(rle), (res, res), interpolation=cv2.INTER_NEAREST)
                if small.any():
                    rles.append(mask_util.encode(np.asfortranarray(small)))
            sets.append(rles)
        out[stem] = sets
    path.write_bytes(pickle.dumps(out))
    return out


def augment(image: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flips, 90-degree rotations and a brightness/gamma jitter; masks follow geometry."""
    k = random.randrange(4)
    image, masks = np.rot90(image, k), np.rot90(masks, k, axes=(1, 2))
    if random.random() < 0.5:
        image, masks = image[:, ::-1], masks[:, :, ::-1]
    image = image.astype(np.float32) / 255.0
    if random.random() < 0.5:
        image = np.clip(image ** random.uniform(0.7, 1.4) * random.uniform(0.9, 1.1), 0, 1)
    return np.ascontiguousarray(image), np.ascontiguousarray(masks)


class DetectionSet(Dataset):
    def __init__(self, indices: np.ndarray, stems: list[str], instances: dict, train: bool):
        self.indices, self.stems, self.instances, self.train = indices, stems, instances, train
        self.images = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self.images is None:
            self.images = np.load(CACHE / "train_images.npy", mmap_mode="r")
        k = self.indices[i]
        rles = random.choice(self.instances[self.stems[k]])
        masks = mask_util.decode(rles).transpose(2, 0, 1) if rles else np.zeros((0, RES, RES), np.uint8)
        image = np.array(self.images[k])
        if self.train:
            image, masks = augment(image, masks)
        else:
            image = image.astype(np.float32) / 255.0
        keep = masks.reshape(len(masks), -1).any(1)
        masks = masks[keep]
        boxes = []
        for m in masks:
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(masks), dtype=torch.int64),
            "masks": torch.from_numpy(masks.astype(np.uint8)),
        }
        return torch.from_numpy(image)[None].repeat(3, 1, 1), target


def build_detector(pretrained: bool = True) -> torch.nn.Module:
    """COCO-pretrained Mask R-CNN v2 with one class and anchors sized for filaments."""
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, min_size=RES, max_size=RES,
                                     box_detections_per_img=100, box_score_thresh=0.05)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 5)
    model.rpn.head.cls_logits = torch.nn.Conv2d(256, 5, 1)
    model.rpn.head.bbox_pred = torch.nn.Conv2d(256, 20, 1)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    return model


def merge_overlapping(rles: list[dict]) -> list[dict]:
    """Union any predicted masks that overlap at all, transitively."""
    if len(rles) < 2:
        return rles
    touching = np.asarray(mask_util.iou(rles, rles, [0] * len(rles))) > 0
    parent = list(range(len(rles)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(*np.nonzero(touching)):
        parent[find(a)] = find(b)
    groups: dict[int, list[dict]] = {}
    for i, rle in enumerate(rles):
        groups.setdefault(find(i), []).append(rle)
    return [mask_util.merge(g) if len(g) > 1 else g[0] for g in groups.values()]


@torch.no_grad()
def detect(model: torch.nn.Module, image: np.ndarray, device: str, min_score: float = 0.2) -> list[tuple[float, dict]]:
    """(score, 2048 px RLE) for each detection above ``min_score`` on one cached image."""
    model.eval()
    x = torch.from_numpy(image.astype(np.float32) / 255.0)[None].repeat(3, 1, 1).to(device)
    out = model([x])[0]
    keep = out["scores"] >= min_score
    scores = out["scores"][keep].cpu().numpy()
    probs = out["masks"][keep, 0].cpu().numpy()
    result = []
    for score, prob in zip(scores, probs):
        full = cv2.resize(prob, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR) > 0.5
        if full.any():
            result.append((float(score), mask_util.encode(np.asfortranarray(full.astype(np.uint8)))))
    return result


def score_detections(dets: dict[str, list], groups: dict, score_thr: float) -> PQCounts:
    total = PQCounts()
    for stem, found in dets.items():
        pred = merge_overlapping([rle for s, rle in found if s >= score_thr])
        for entry in groups[stem]:
            total += entry_counts(entry.rles, pred)
    return total


def collate(batch):
    return tuple(zip(*batch))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--name", default="mrcnn")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=2)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
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

    stems = json.loads((CACHE / "index.json").read_text())["train"]
    instances = instance_cache()
    if args.fold >= 0:
        train_idx, val_idx = fold_split(stems, args.fold)
    else:
        train_idx, val_idx = np.arange(len(stems)), np.array([], dtype=int)
    groups = group_by_stem(load_train()) if len(val_idx) else {}
    all_images = np.load(CACHE / "train_images.npy", mmap_mode="r")
    emit(f"device={device} train={len(train_idx)} val={len(val_idx)} args={vars(args)}")

    loader = DataLoader(DetectionSet(train_idx, stems, instances, train=True), batch_size=args.batch,
                        shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0,
                        collate_fn=collate, drop_last=True)
    model = build_detector().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    steps, warmup = args.epochs * len(loader), min(500, len(loader))
    milestones = (int(steps * 0.7), int(steps * 0.9))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: min(1.0, (s + 1) / warmup) * (0.1 ** sum(s >= m for m in milestones)))

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for step, (images, targets) in enumerate(loader, start=1):
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                emit(f"non-finite loss at epoch {epoch} step {step}; stopping")
                raise SystemExit(1)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            optimiser.step()
            schedule.step()
            running += loss.item()
            if step % 50 == 0:
                rate = (time.time() - started) / step
                emit(f"  epoch {epoch} step {step}/{len(loader)} loss={running / step:.4f} {rate:.2f}s/step")
        emit(f"epoch {epoch} loss={running / len(loader):.4f} {time.time() - started:.0f}s")

        if len(val_idx) and (epoch % args.val_every == 0 or epoch == args.epochs):
            dets = {stems[k]: detect(model, np.array(all_images[k]), device) for k in val_idx}
            scored = {thr: score_detections(dets, groups, thr) for thr in (0.3, 0.5, 0.7)}
            for thr, counts in scored.items():
                emit(f"  val score>={thr}: {counts.summary()}")
            top = max(c.score() for c in scored.values())
            if top > best:
                best = top
                torch.save(model.state_dict(), run_dir / "best.pt")
                (run_dir / "val_dets.pkl").write_bytes(pickle.dumps(dets))
    torch.save(model.state_dict(), run_dir / "last.pt")
    emit(f"done best_val_pq={best:.4f}")


if __name__ == "__main__":
    main()
