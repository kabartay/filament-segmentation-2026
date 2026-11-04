"""Post-processing variants on 5-fold OOF probabilities: hysteresis, size-dependent confidence; near-miss diagnosis."""
import json, itertools, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from pycocotools import mask as mu
from filseg.cache import CACHE
from filseg.data import SIZE, group_by_stem, load_train
from filseg.geometry import fit_disk, longitude_deg
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import upsample
from filseg.train import RUNS

G = {}
def init():
    G["groups"] = group_by_stem(load_train())
    G["index"] = json.loads((CACHE / "index.json").read_text())["train"]
    G["images"] = np.load(CACHE / "train_images.npy", mmap_mode="r")
    G["probs"] = {f: np.load(RUNS / f"unet_r34full_f{f}" / "val_probs.npy", mmap_mode="r") for f in range(5)}

def comps(full, thr, close_px, disk, min_area=150):
    binary = (full > thr).astype(np.uint8)
    grouping = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))) if close_px else binary
    n, labels = cv2.connectedComponents(grouping, connectivity=8)
    labels[binary == 0] = 0
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=n); sums = np.bincount(flat, weights=full.ravel(), minlength=n)
    mx = np.zeros(n); np.maximum.at(mx, flat, full.ravel())
    ys, xs = np.divmod(np.arange(flat.size), SIZE)
    cx = np.bincount(flat, weights=xs, minlength=n) / np.maximum(areas, 1); cy = np.bincount(flat, weights=ys, minlength=n) / np.maximum(areas, 1)
    lons = longitude_deg(cx, cy, disk)
    out = []
    for l in range(1, n):
        if areas[l] < min_area: continue
        out.append(dict(rle=mu.encode(np.asfortranarray((labels == l).astype(np.uint8))), area=int(areas[l]), mean=sums[l] / areas[l], max=mx[l], lon=lons[l]))
    return out

BASE = [("base", a, s, None) for a in (200, 300) for s in (0.8, 0.85)]
SIZEDEP = [("size", a, (ss, sl), sp) for a in (150, 200, 300) for sp in (800, 1500, 3000) for ss in (0.85, 0.9) for sl in (0.7, 0.75, 0.8)]
HYST_T = (0.3, 0.35, 0.4)
HYST = [("hyst", a, (mean, mx), None) for a in (200, 300, 400) for mean in (0.0, 0.7, 0.75, 0.8) for mx in (0.0, 0.85, 0.9, 0.95)]

def select(cs, kind, area, score, split):
    if kind == "base": return [i for i, c in enumerate(cs) if c["area"] >= area and c["mean"] >= score]
    if kind == "size": return [i for i, c in enumerate(cs) if c["area"] >= area and c["mean"] >= (score[0] if c["area"] < split else score[1])]
    return [i for i, c in enumerate(cs) if c["area"] >= area and c["mean"] >= score[0] and c["max"] >= score[1]]

def work(item):
    fold, i, stem = item
    full = upsample(np.asarray(G["probs"][fold][i], np.float32))
    disk = fit_disk(np.array(G["images"][G["index"].index(stem)]))
    ents = G["groups"][stem]
    res = defaultdict(PQCounts); diag = []
    variants = {("t", 0.45): comps(full, 0.45, 31, disk)}
    for t in HYST_T: variants[("t", t)] = comps(full, t, 31, disk)
    for (_, t), cs in variants.items():
        rles = [c["rle"] for c in cs]
        mats = [(len(e.rles), iou_matrix(e.rles, rles), e) for e in ents]
        cfgs = BASE + SIZEDEP if t == 0.45 else []
        cfgs = cfgs + HYST
        for cfg in cfgs:
            cols = select(cs, *cfg)
            for n, m, _ in mats: res[(t,) + tuple(map(str, cfg))].add(m[:, cols], n, len(cols))
        if t == 0.45:  # diagnosis at the submission setting
            cols = select(cs, "base", 300, 0.8, None)
            for n, m, e in mats:
                if n == 0 or not cols: continue
                sub = m[:, cols]; areas = mu.area(e.rles)
                for k in range(n):
                    j = sub[k].argmax(); best = sub[k, j]
                    if 0.3 <= best <= 0.5:
                        pa = cs[cols[j]]["area"]; gts_hit = int((sub[:, j] > 0.1).sum()); preds_hit = int((sub[k] > 0.1).sum())
                        diag.append((pa / areas[k], gts_hit, preds_hit))
    return dict(res), diag

if __name__ == "__main__":
    items = []
    for f in range(5):
        for i, s in enumerate(json.loads((RUNS / f"unet_r34full_f{f}" / "val_stems.json").read_text())): items.append((f, i, s))
    total = defaultdict(PQCounts); diag = []
    with ProcessPoolExecutor(4, initializer=init) as ex:
        for r, d in ex.map(work, items, chunksize=4):
            for k, v in r.items(): total[k] += v
            diag += d
    ranked = sorted(total.items(), key=lambda kv: -kv[1].score())
    for kind in ("base", "size", "hyst"):
        print(f"--- {kind}")
        for k, v in [kv for kv in ranked if kv[0][1] == kind][:6]: print(k, v.summary())
    d = np.array(diag)
    ratio = d[:, 0]
    print(f"near-miss FN {len(d)}: pred/GT area ratio <0.67 {(ratio<0.67).mean():.2f}, 0.67-1.5 {((ratio>=0.67)&(ratio<=1.5)).mean():.2f}, >1.5 {(ratio>1.5).mean():.2f}")
    print(f"  best pred also touches other GTs (merge) {(d[:,1]>1).mean():.2f}; GT touched by several preds (split) {(d[:,2]>1).mean():.2f}")
