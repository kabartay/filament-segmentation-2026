"""U-Net components + Mask R-CNN instances on fold-0 validation; all outputs disjoint."""
import json, pickle, sys
import cv2
import numpy as np
from pycocotools import mask as mask_util
sys.path.insert(0, ".")
from filseg.cache import CACHE
from filseg.data import SIZE, group_by_stem, load_train
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, entry_counts
from filseg.predict import load_post
from filseg.postprocess import instances, upsample
from filseg.train import RUNS

def enc(m): return mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
def dec(r): return mask_util.decode(r).astype(bool)

dets = pickle.loads((RUNS / "mrcnn_f0" / "val_dets.pkl").read_bytes())
groups = group_by_stem(load_train())
stems = json.loads((RUNS / "sub1" / "val_stems.json").read_text())
probs = np.load(RUNS / "sub1" / "val_probs.npy", mmap_mode="r")
index = json.loads((CACHE / "index.json").read_text())["train"]
images = np.load(CACHE / "train_images.npy", mmap_mode="r")
cfg = load_post(RUNS / "sub1" / "post.json")
assert set(stems) == set(dets)

def paint(found, thr, frac=0.5):
    taken, out = np.zeros((SIZE, SIZE), bool), []
    for s, rle in sorted([d for d in found if d[0] >= thr], key=lambda d: -d[0]):
        m = dec(rle); own = m & ~taken
        if own.any() and own.sum() >= frac * m.sum():
            out.append(own); taken |= own
    return out

variants = {k: PQCounts() for k in ("unet", "det", "unet+det_extra", "det_px_unet", "unet_split_by_det")}
for k, stem in enumerate(stems):
    prob = np.asarray(probs[k], np.float32)
    disk = fit_disk(np.array(images[index.index(stem)]))
    unet = [dec(r) for r in instances(prob, cfg, disk)]
    det = paint(dets[stem], 0.75)
    fg = upsample(prob) >= cfg.threshold
    out = {"unet": unet, "det": det}
    # 1. add detector instances that touch no U-Net component
    union = np.logical_or.reduce(unet) if unet else np.zeros((SIZE, SIZE), bool)
    out["unet+det_extra"] = unet + [d for d in det if not (d & union).any()]
    # 2. detector grouping, U-Net pixels: dilated detector mask intersected with the U-Net foreground
    kernel = np.ones((7, 7), np.uint8)
    taken, px = np.zeros((SIZE, SIZE), bool), []
    for d in det:
        m = cv2.dilate(d.astype(np.uint8), kernel).astype(bool) & fg & ~taken
        if m.sum() >= 200: px.append(m); taken |= m
    out["det_px_unet"] = px
    # 3. split a U-Net component when >= 2 detector instances each cover >= 25% of it
    split = []
    for u in unet:
        inside = [(d & u) for d in det if (d & u).sum() >= 0.25 * u.sum()]
        if len(inside) < 2:
            split.append(u); continue
        seeds = np.zeros(u.shape, np.int32)
        for i, m in enumerate(inside, 1): seeds[m] = i
        # assign each pixel of u to its nearest seed
        dist = [cv2.distanceTransform((~(seeds == i)).astype(np.uint8), cv2.DIST_L2, 3) for i in range(1, len(inside) + 1)]
        lab = np.argmin(np.stack(dist), 0) + 1
        split += [u & (lab == i) for i in range(1, len(inside) + 1) if (u & (lab == i)).any()]
    out["unet_split_by_det"] = split
    for name, masks in out.items():
        rles = [enc(m) for m in masks]
        for e in groups[stem]:
            variants[name] += entry_counts(e.rles, rles)
for name, c in variants.items():
    print(f"{name:20s} {c.summary()}")
