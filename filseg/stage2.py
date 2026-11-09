"""Stage 2: re-segment every selected filament in its own rotated crop at native resolution.

Stage 1 (``filseg.tree``) decides which filaments exist and gives each one a seed mask, a component of the
probability map at some threshold. A level set of one map cannot represent every filament: a fragment may be
selected, or a filament merges with a neighbour at the level where it would be complete. Stage 2 redraws each
seed from a crop around it:

* **Crop.** The seed's principal axis is turned horizontal and the crop covers 1.5 times its length, so it
  shows where the filament ends; it is scaled down only when it does not fit ``CROP_W x CROP_H``, never up.
* **Inputs** (``CHANNELS``): the image, the stage-1 probability, the seed, and the other selected seeds of
  the image (where a neighbour begins).
* **Target.** The mean of the masks drawn for this filament by the annotators who drew it (the annotation of
  each annotator entry that overlaps the seed most, if that IoU exceeds ``DREW_IOU``). Whether the filament
  exists is stage 1's decision, so annotators who did not draw it do not enter the target.
* **Paste-back.** The output is thresholded at ``T_PIX`` and only the pieces touching the seed are kept; a seed
  covered at least half by other instances' refined masks is absorbed; overlaps go to the instance with the
  higher refined probability, so the masks stay disjoint. Seeds whose crop had to be scaled below ``MIN_SCALE``
  are kept as they are.

The idea follows the stage-2 refiner of the public notebook "Solar Filament Segmentation 2026 (0.40)"
(tushhaaarrrr); this is an independent implementation on our stage 1. Results: docs/DESIGN.md, "Stage 2".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from pycocotools import mask as mask_util

from filseg.data import SIZE

CROP_W, CROP_H = 384, 128
CHANNELS = ("image", "prob", "seed", "others")
LENGTH_PAD = 1.5  # crop length / seed length along the axis
WIDTH_PAD = 1.5  # crop height / seed width, plus WIDTH_MARGIN pixels
WIDTH_MARGIN = 32
MIN_SCALE = 0.5
T_PIX = 0.40
DREW_IOU = 0.3


@dataclass
class Frame:
    """Affine map from the full image to one crop: crop = M @ [x, y, 1]; ``scale`` <= 1."""

    M: np.ndarray  # (2, 3) float64
    scale: float


def frame_of(seed: np.ndarray) -> Frame:
    """Crop frame of a seed mask: principal axis horizontal, centred, scaled down only if needed."""
    ys, xs = np.nonzero(seed)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(axis=0)
    if len(pts) > 1:
        _, vecs = np.linalg.eigh(np.cov((pts - mean).T))
        d = vecs[:, -1]
    else:
        d = np.array([1.0, 0.0])
    n = np.array([-d[1], d[0]])
    a, b = (pts - mean) @ d, (pts - mean) @ n
    length, width = a.max() - a.min() + 1, b.max() - b.min() + 1
    centre = mean + d * (a.max() + a.min()) / 2 + n * (b.max() + b.min()) / 2
    scale = min(1.0, CROP_W / (LENGTH_PAD * length), CROP_H / (WIDTH_PAD * width + WIDTH_MARGIN))
    M = np.array([[scale * d[0], scale * d[1], 0.0], [scale * n[0], scale * n[1], 0.0]])
    M[:, 2] = np.array([CROP_W / 2, CROP_H / 2]) - M[:, :2] @ centre
    return Frame(M, float(scale))


def warp(full: np.ndarray, frame: Frame, nearest: bool = False) -> np.ndarray:
    """The crop of a full-size (SIZE, SIZE) array."""
    flag = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(full, frame.M, (CROP_W, CROP_H), flags=flag, borderValue=0)


def unwarp(crop: np.ndarray, frame: Frame) -> np.ndarray:
    """A crop mapped back to (SIZE, SIZE); zero outside the crop."""
    return cv2.warpAffine(crop.astype(np.float32), frame.M, (SIZE, SIZE),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)


def crop_inputs(image: np.ndarray, prob: np.ndarray, seeds: list[np.ndarray]) -> tuple[np.ndarray, list[Frame]]:
    """(n, 4, CROP_H, CROP_W) uint8 inputs of every seed of one image, and their frames.

    Args:
        image: (SIZE, SIZE) uint8 raw image.
        prob: (SIZE, SIZE) float32 stage-1 probability.
        seeds: boolean (SIZE, SIZE) masks of the selected instances.
    """
    union = np.zeros((SIZE, SIZE), bool)
    for s in seeds:
        union |= s
    prob8 = to_u8(prob)
    out = np.zeros((len(seeds), len(CHANNELS), CROP_H, CROP_W), np.uint8)
    frames = []
    for i, s in enumerate(seeds):
        out[i], fr = crop_one(image, prob8, s, union & ~s)
        frames.append(fr)
    return out, frames


def to_u8(prob: np.ndarray) -> np.ndarray:
    """A probability map in [0, 1] as uint8, the form ``crop_one`` takes."""
    return np.clip(prob * 255 + 0.5, 0, 255).astype(np.uint8)


def crop_one(image: np.ndarray, prob8: np.ndarray, seed: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, Frame]:
    """(4, CROP_H, CROP_W) uint8 input of one seed and its frame.

    Args:
        image: (SIZE, SIZE) uint8 raw image.
        prob8: (SIZE, SIZE) uint8 stage-1 probability (``to_u8``).
        seed: boolean (SIZE, SIZE) mask to redraw.
        others: boolean (SIZE, SIZE) mask of the neighbouring instances (not overlapping ``seed``).
    """
    fr = frame_of(seed)
    x = np.stack([warp(image, fr), warp(prob8, fr), warp(seed.astype(np.uint8) * 255, fr),
                  warp(others.astype(np.uint8) * 255, fr)])
    return x, fr


def crop_target(seed: np.ndarray, entries: list[list[dict]], frame: Frame,
                contain: float | None = None) -> tuple[np.ndarray, int]:
    """Soft target crop (CROP_H, CROP_W) uint8 and the number of annotator entries that drew the filament.

    Args:
        seed: boolean (SIZE, SIZE) seed mask.
        entries: per annotator entry, its filament RLEs.
        frame: the seed's crop frame.
        contain: if set, an entry's best-IoU mask also counts as drawing the filament when it holds at least
            this share of the seed's area (the seed is a fragment of it), whatever the IoU.
    """
    seed_rle = mask_util.encode(np.asfortranarray(seed.astype(np.uint8)))
    seed_area = float(mask_util.area(seed_rle))
    acc = np.zeros((CROP_H, CROP_W), np.float32)
    drew = 0
    for rles in entries:
        if not rles:
            continue
        iou = np.asarray(mask_util.iou(rles, [seed_rle], [0])).ravel()
        k = int(iou.argmax())
        inside = 0.0
        if contain is not None and iou[k] > 0:
            area_k = float(mask_util.area(rles[k]))
            inside = iou[k] * (area_k + seed_area) / (1 + iou[k]) / seed_area  # |seed & mask| / |seed|
        if iou[k] > DREW_IOU or (contain is not None and inside >= contain):
            acc += warp(mask_util.decode(rles[k]).astype(np.float32), frame)
            drew += 1
    target = acc / drew if drew else acc
    return np.clip(target * 255 + 0.5, 0, 255).astype(np.uint8), drew


def build_refiner(arch: str = "Unet", encoder: str = "resnet34", pretrained: bool = True) -> torch.nn.Module:
    """A segmentation network over the ``CHANNELS`` crop inputs."""
    return getattr(smp, arch)(encoder, encoder_weights="imagenet" if pretrained else None,
                              in_channels=len(CHANNELS), classes=1)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """uint8 (n, 4, H, W) crops to normalised float inputs."""
    t = torch.from_numpy(np.ascontiguousarray(x)).float() / 255.0
    t[:, 0] = (t[:, 0] - 0.5) / 0.25  # the image, normalised as in filseg.train.normalise
    return t


@torch.no_grad()
def predict_crops(models: list[torch.nn.Module], x: np.ndarray, device: str, batch: int = 32) -> np.ndarray:
    """Mean sigmoid over models and 4 flips, (n, CROP_H, CROP_W) float32."""
    out = np.zeros((len(x), CROP_H, CROP_W), np.float32)
    views = [(), (-1,), (-2,), (-2, -1)]
    for lo in range(0, len(x), batch):
        t = to_tensor(x[lo:lo + batch]).to(device)
        acc = torch.zeros(t.shape[0], 1, CROP_H, CROP_W, device=device)
        for m in models:
            m.eval()
            for d in views:
                acc += torch.sigmoid(m(t.flip(d))).flip(d) if d else torch.sigmoid(m(t))
        out[lo:lo + batch] = (acc / (len(models) * len(views)))[:, 0].float().cpu().numpy()
    return out


def load_refiners(paths: list, device: str, arch: str = "Unet", encoder: str = "resnet34") -> list[torch.nn.Module]:
    """Refiner checkpoints (float16 or float32 state dicts) as float32 models in eval mode."""
    models = []
    for p in paths:
        m = build_refiner(arch, encoder, pretrained=False)
        state = torch.load(p, map_location="cpu")
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
        models.append(m.to(device).eval())
    return models


def refine(models: list[torch.nn.Module], image: np.ndarray, prob: np.ndarray, rles: list[dict],
           score: np.ndarray, device: str, t_pix: float = T_PIX) -> list[dict]:
    """Stage 2 on one image: the selected instances redrawn by the refiners, as disjoint COCO RLEs.

    Args:
        models: refiners (``load_refiners``); their outputs are averaged over models and four flips.
        image: (SIZE, SIZE) uint8 raw image.
        prob: (SIZE, SIZE) float32 stage-1 probability (the mean of the flipped views, upsampled).
        rles: stage-1 selected instances (``TreeScorer.select_scored``).
        score: their stage-1 scores.
        device: torch device of the models.
        t_pix: pixel threshold of the refined probability.
    """
    if not rles:
        return []
    seeds = [mask_util.decode(r).astype(bool) for r in rles]
    x, frames = crop_inputs(image, prob, seeds)
    pred = predict_crops(models, x, device)
    refined = [None if fr.scale < MIN_SCALE else unwarp(q, fr) for q, fr in zip(pred, frames)]
    masks = paste_back(seeds, refined, np.asarray(score), t_pix)
    return [mask_util.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]


def paste_back(seeds: list[np.ndarray], refined: list[np.ndarray | None], score: np.ndarray,
               t_pix: float = T_PIX) -> list[np.ndarray]:
    """Disjoint final masks of one image.

    Args:
        seeds: boolean (SIZE, SIZE) stage-1 masks.
        refined: per seed, its (SIZE, SIZE) refined probability, or None to keep the seed.
        score: stage-1 scores; the lower-scored of two seeds is absorbed first.
        t_pix: pixel threshold of the refined probability.

    Returns:
        boolean masks, one per surviving instance, pairwise disjoint.
    """
    n = len(seeds)
    if n == 0:
        return []
    prob = []
    masks = []
    for s, q in zip(seeds, refined):
        if q is None:
            prob.append(s.astype(np.float32))
            masks.append(s.copy())
            continue
        m = q >= t_pix
        n_lab, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        touch = np.unique(lab[s & m])
        touch = touch[touch > 0]
        m = np.isin(lab, touch) if len(touch) else np.zeros_like(m)
        if m.sum() == 0:  # the refiner erased the seed: keep stage 1's mask
            prob.append(s.astype(np.float32))
            masks.append(s.copy())
        else:
            prob.append(np.where(m, q, 0).astype(np.float32))
            masks.append(m)
    alive = np.ones(n, bool)
    for i in np.argsort(score):  # absorb seeds covered by the others' refined masks, weakest first
        others = np.zeros((SIZE, SIZE), bool)
        for j in range(n):
            if j != i and alive[j]:
                others |= masks[j]
        if (seeds[i] & others).sum() >= 0.5 * seeds[i].sum():
            alive[i] = False
    keep = np.nonzero(alive)[0]
    if len(keep) == 0:
        return []
    stack = np.stack([np.where(masks[i], prob[i], -1.0) for i in keep])
    owner = stack.argmax(axis=0)
    covered = stack.max(axis=0) >= 0
    out = []
    for r, i in enumerate(keep):
        m = covered & (owner == r)
        if m.sum():
            out.append(m)
    return out
