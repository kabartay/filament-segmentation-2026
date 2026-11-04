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
stems = json.loads((RUNS / "unet_r34full_f0" / "val_stems.json").read_text())
probs = np.load(RUNS / "unet_r34full_f0" / "val_probs.npy", mmap_mode="r")
index = json.loads((CACHE / "index.json").read_text())["train"]
images = np.load(CACHE / "train_images.npy", mmap_mode="r")
cfg = load_post(RUNS / "unet_r34full_f0" / "post.json")
assert set(stems) == set(dets)

def paint(found, thr, frac=0.5):
    taken, out = np.zeros((SIZE, SIZE), bool), []
    for s, rle in sorted([d for d in found if d[0] >= thr], key=lambda d: -d[0]):
        m = dec(rle); own = m & ~taken
        if own.any() and own.sum() >= frac * m.sum():
            out.append(own); taken |= own
    return out

variants = {k: PQCounts() for k in ("unet", "unet+det_extra75", "unet+det_extra85", "unet+det_extra90")}
for k, stem in enumerate(stems):
    prob = np.asarray(probs[k], np.float32)
    disk = fit_disk(np.array(images[index.index(stem)]))
    unet = [dec(r) for r in instances(prob, cfg, disk)]
    det = paint(dets[stem], 0.75)
    fg = upsample(prob) >= cfg.threshold
    union = np.logical_or.reduce(unet) if unet else np.zeros((SIZE, SIZE), bool)
    out = {"unet": unet}
    for t in (0.75, 0.85, 0.9):
        out[f"unet+det_extra{int(t*100)}"] = unet + [d for d in paint(dets[stem], t) if not (d & union).any()]
    for name, masks in out.items():
        rles = [enc(m) for m in masks]
        for e in groups[stem]:
            variants[name] += entry_counts(e.rles, rles)
for name, c in variants.items():
    print(f"{name:20s} {c.summary()}")
