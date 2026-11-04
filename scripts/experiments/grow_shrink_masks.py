"""Do predicted masks sit too tight? Grow/shrink each kept component by k px, or grow it into nearby pixels above a lower threshold."""
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from pycocotools import mask as mu
from filseg.cache import CACHE
from filseg.data import group_by_stem, load_train
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import PostConfig, components, keep, upsample
from filseg.train import RUNS
G = {}
def init():
    G["groups"] = group_by_stem(load_train()); G["index"] = json.loads((CACHE/"index.json").read_text())["train"]
    G["images"] = np.load(CACHE/"train_images.npy", mmap_mode="r")
    G["probs"] = {f: np.load(RUNS/f"unet_r34full_f{f}"/"val_probs.npy", mmap_mode="r") for f in range(5)}
CFG = PostConfig(0.45, 31, 300, 0.85, 90.0)
VARIANTS = [("k", k) for k in (-2, -1, 0, 1, 2, 3, 5)] + [("grow", t, d) for t in (0.2, 0.3, 0.35) for d in (3, 7)]
def work(item):
    f, i, s = item
    full = upsample(np.asarray(G["probs"][f][i], np.float32))
    cs = components(full, CFG.threshold, CFG.close_px, fit_disk(np.array(G["images"][G["index"].index(s)])), CFG.min_area)
    cs = [cs[j] for j in keep(cs, CFG)]
    order = np.argsort([-c.score for c in cs]); masks = [mu.decode(cs[j].rle) for j in order]
    res = {}
    for v in VARIANTS:
        taken = np.zeros_like(full, np.uint8); rles = []
        for m in masks:
            if v[0] == "k":
                k = v[1]
                if k > 0: m2 = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1)))
                elif k < 0: m2 = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-2*k+1, -2*k+1)))
                else: m2 = m
            else:
                _, t, d = v
                m2 = (m | (cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*d+1, 2*d+1))) & (full > t))).astype(np.uint8)
            m2 = (m2 & (1 - taken)).astype(np.uint8)
            if m2.sum() == 0: continue
            taken |= m2; rles.append(mu.encode(np.asfortranarray(m2)))
        c = PQCounts()
        for e in G["groups"][s]: c.add(iou_matrix(e.rles, rles), len(e.rles), len(rles))
        res[v] = c
    return res
if __name__ == "__main__":
    items = [(f, i, s) for f in range(5) for i, s in enumerate(json.loads((RUNS/f"unet_r34full_f{f}"/"val_stems.json").read_text()))]
    tot = defaultdict(PQCounts)
    with ProcessPoolExecutor(4, initializer=init) as ex:
        for r in ex.map(work, items, chunksize=4):
            for k, v in r.items(): tot[k] += v
    for k in VARIANTS: print(k, tot[k].summary())
