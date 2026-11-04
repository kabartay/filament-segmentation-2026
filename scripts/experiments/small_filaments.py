"""Inspect small GT filaments: annotator agreement, model probability, why they are missed; FP agreement."""
import json
from collections import Counter
import numpy as np, cv2
from pycocotools import mask as mu
from filseg.cache import CACHE
from filseg.data import group_by_stem, load_train, read_image, TRAIN_IMAGES
from filseg.geometry import fit_disk
from filseg.metric import iou_matrix
from filseg.postprocess import PostConfig, components, keep, upsample
from filseg.train import RUNS

groups = group_by_stem(load_train()); index = json.loads((CACHE/"index.json").read_text())["train"]
images = np.load(CACHE/"train_images.npy", mmap_mode="r"); cfg = PostConfig(0.45, 31, 300, 0.8, 90.0)

# 1. agreement by size on multi-annotator images (other annotator has a GT with IoU > 0.5 / any overlap)
bins = [0, 300, 500, 1000, 2000, 5000, 10**9]
agree = {b: [] for b in range(len(bins)-1)}
for s, g in groups.items():
    if len(g) < 2: continue
    for a in g:
        if not a.rles: continue
        areas = mu.area(a.rles)
        others = [b for b in g if b is not a]
        for k, area in enumerate(areas):
            hits = sum(1 for b in others if b.rles and iou_matrix([a.rles[k]], b.rles).max() > 0.5)
            agree[np.digitize(area, bins) - 1].append(hits / len(others))
print("share of the OTHER annotators who mark the same filament (IoU>0.5), by GT area:")
for b, v in agree.items(): print(f"  {bins[b]}-{bins[b+1]}: n={len(v)}, mean agreement {np.mean(v):.2f}, none agree {np.mean(np.array(v)==0):.2f}")

# 2. why small GTs are missed (OOF, 5 folds)
why = Counter(); maxp = []; examples = []
fp_agree = Counter()
for f in range(5):
    stems = json.loads((RUNS/f"unet_r34full_f{f}"/"val_stems.json").read_text()); probs = np.load(RUNS/f"unet_r34full_f{f}"/"val_probs.npy", mmap_mode="r")
    for i, s in enumerate(stems):
        full = upsample(np.asarray(probs[i], np.float32))
        cs = components(full, 0.45, 31, fit_disk(np.array(images[index.index(s)])), min_area=1)
        kept = keep(cs, cfg); allr = [c.rle for c in cs]; kr = [allr[j] for j in kept]
        for e in groups[s]:
            if not e.rles: continue
            ia = iou_matrix(e.rles, allr); ik = iou_matrix(e.rles, kr) if kr else np.zeros((len(e.rles), 0))
            for k, area in enumerate(mu.area(e.rles)):
                if area >= 500: continue
                if ik.shape[1] and ik[k].max() > 0.5: why["TP"] += 1; continue
                m = mu.decode(e.rles[k]).astype(bool); mp = full[m].max(); maxp.append(mp)
                if ia.shape[1] and ia[k].max() > 0.5:
                    c = cs[int(ia[k].argmax())]
                    why["found, removed by " + ("area" if c.area < 300 else "confidence")] += 1
                elif mp < 0.45: why["prob below threshold"] += 1
                elif ia.shape[1] and ia[k].max() > 0: why["covered but merged/wrong extent"] += 1
                else: why["other"] += 1
                if len(examples) < 16 and mp < 0.45 and i % 3 == 0: examples.append((s, e.rles[k], full))
print("small GT (<500 px) outcomes:", dict(why))
mp = np.array(maxp); print("max prob inside missed small GT: quartiles", np.round(np.percentile(mp, [25, 50, 75]), 2).tolist(), " <0.2:", round((mp < 0.2).mean(), 2))

# 3. panel of missed small filaments: image crop with GT contour (green) and prob contour 0.3 (red)
tiles = []
for s, rle, full in examples[:16]:
    img = read_image(TRAIN_IMAGES / f"{s}.jpeg"); m = mu.decode(rle)
    ys, xs = np.nonzero(m); cy, cx = int(ys.mean()), int(xs.mean()); h = 96
    y0, x0 = max(cy-h, 0), max(cx-h, 0)
    crop = cv2.cvtColor(img[y0:y0+2*h, x0:x0+2*h], cv2.COLOR_GRAY2BGR)
    for mask, col in ((m, (0, 255, 0)), ((full > 0.3).astype(np.uint8), (0, 0, 255))):
        cnts, _ = cv2.findContours(np.ascontiguousarray(mask[y0:y0+2*h, x0:x0+2*h]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, cnts, -1, col, 1)
    tiles.append(cv2.resize(cv2.copyMakeBorder(crop, 0, 2*h-crop.shape[0], 0, 2*h-crop.shape[1], cv2.BORDER_CONSTANT), (256, 256), interpolation=cv2.INTER_NEAREST))
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[r:r+4]) for r in range(0, len(tiles), 4)]
cv2.imwrite("docs/figures/missed_small.png", np.vstack(rows)); print("wrote docs/figures/missed_small.png")
