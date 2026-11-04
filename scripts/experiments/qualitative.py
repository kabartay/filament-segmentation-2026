"""Qualitative figure for the report: out-of-fold predictions next to every annotator's masks.

Picks fold-0 validation images with 3 annotators, draws each annotator's filaments in its own colour
and the prediction (submission setting) in red, and writes docs/figures/qualitative.png.
Run: PYTHONPATH=. .venv/bin/python scripts/experiments/qualitative.py
"""
import json
import cv2
import numpy as np
from pycocotools import mask as mu
from filseg.cache import CACHE
from filseg.data import TRAIN_IMAGES, group_by_stem, load_train, read_image
from filseg.geometry import fit_disk
from filseg.metric import PQCounts, iou_matrix
from filseg.postprocess import PostConfig, instances
from filseg.train import RUNS

CFG = PostConfig(0.45, 31, 300, 0.8, 90.0)
COLOURS = [(0, 200, 0), (0, 200, 255), (255, 160, 0)]  # BGR: annotators
groups = group_by_stem(load_train())
index = json.loads((CACHE / "index.json").read_text())["train"]
images = np.load(CACHE / "train_images.npy", mmap_mode="r")
stems = json.loads((RUNS / "unet_r34full_f0" / "val_stems.json").read_text())
probs = np.load(RUNS / "unet_r34full_f0" / "val_probs.npy", mmap_mode="r")

scored = []
for i, s in enumerate(stems):
    if len(groups[s]) != 3:
        continue
    pred = instances(np.asarray(probs[i], np.float32), CFG, fit_disk(np.array(images[index.index(s)])))
    c = PQCounts()
    for e in groups[s]:
        c.add(iou_matrix(e.rles, pred), len(e.rles), len(pred))
    scored.append((c.score(), s, pred))
scored.sort()
picks = [scored[len(scored) // 2], scored[(3 * len(scored)) // 4], scored[-2]]  # median, upper quartile, near best

panels = []
for pq, s, pred in picks:
    img = cv2.cvtColor(read_image(TRAIN_IMAGES / f"{s}.jpeg"), cv2.COLOR_GRAY2BGR)
    for e, col in zip(groups[s], COLOURS):
        for r in e.rles:
            cnts, _ = cv2.findContours(mu.decode(r), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, col, 3)
    for r in pred:
        cnts, _ = cv2.findContours(mu.decode(r), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, (40, 40, 255), 5)
    img = cv2.resize(img[160:1888, 160:1888], (900, 900), interpolation=cv2.INTER_AREA)
    cv2.putText(img, f"{s}  PQ {pq:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    panels.append(img)
cv2.imwrite("docs/figures/qualitative.png", np.hstack(panels))
print([(round(p, 3), s) for p, s, _ in picks])
