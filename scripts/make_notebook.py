"""Generate pipeline.ipynb from the cells below (run from the repository root).

Then execute it with: jupyter nbconvert --to notebook --execute --inplace pipeline.ipynb
"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Solar filament instance segmentation: full pipeline

Kaggle *Solar Filament Segmentation Challenge 2026* (MAGFiLO, GONG H-alpha, 2048 x 2048 px).
This notebook reproduces the submitted predictions from the raw test images and the weights stored in
this repository (`weights/unet_r34_all.pt`); no training is needed. Training, tuning and the
validation study are described at the end and can be re-run with the commands given there.

**Pipeline in one line:** downsample to 1024 px -> U-Net (ResNet-34 encoder, 1 input channel) with a
horizontal-flip average -> probability map upsampled to 2048 px -> threshold 0.45 -> closing (31 px,
used only to group nearby fragments) -> 8-connected components -> keep components of at least 300 px
whose mean probability is at least 0.8 -> one compressed COCO RLE per filament.

Every design choice, with the experiments behind it (including the failed ones), is in `docs/DESIGN.md`.
""")
md("## 1. Setup")
code(r"""
import json, os, time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pycocotools import mask as mask_util

from filseg.cache import downsample
from filseg.data import ROOT, SIZE, TEST_IMAGES, read_image
from filseg.geometry import fit_disk
from filseg.postprocess import PostConfig, instances
from filseg.predict import write_csv
from filseg.train import build_model, predict

# FILSEG_DEVICE overrides the automatic choice, e.g. FILSEG_DEVICE=cpu
DEVICE = os.environ.get("FILSEG_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
WEIGHTS = ROOT / "weights" / "unet_r34_all.pt"
OUTPUT = ROOT / "submissions" / "notebook_submission.csv"
RES = 1024  # network input resolution
POST = PostConfig(threshold=0.45, close_px=31, min_area=300, min_score=0.8, max_lon=90.0)
print("device:", DEVICE, "| test images:", TEST_IMAGES.relative_to(ROOT))
""")
md(r"""
## 2. Data

Test images are read from `data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images` (the competition's
download, unzipped into `data/raw/`). Only these H-alpha images are used at inference.
""")
code(r"""
test_paths = sorted(TEST_IMAGES.glob("*.jpeg"))
print(len(test_paths), "test images; first:", test_paths[0].name)
example = read_image(test_paths[0])
plt.figure(figsize=(6, 6)); plt.imshow(example, cmap="gray"); plt.axis("off"); plt.title(test_paths[0].stem); plt.show()
""")
md(r"""
## 3. Model

`segmentation_models_pytorch` U-Net with an ImageNet-initialised ResNet-34 encoder adapted to one
input channel, trained on all 707 training images for 40 epochs (BCE + soft Dice on the consensus
target: the fraction of annotators marking each pixel). The weights are loaded here; nothing is trained.
""")
code(r"""
model = build_model("resnet34", pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")
""")
md(r"""
## 4. One image, step by step

Preprocessing is an area downsample to 1024 px and a fixed normalisation (inside `predict`). The
probability is the average over the image and its horizontal flip. Post-processing turns the map into
disjoint instances; the solar disk is fitted with a Hough circle on the limb (used by the optional
limb filter, which is off: `max_lon=90`).
""")
code(r"""
small = downsample(example, RES)
prob = predict(model, small[None], DEVICE)[0].astype(np.float32)
disk = fit_disk(small)
rles = instances(prob, POST, disk)
print(f"{len(rles)} filaments; disk centre ({disk.cx:.0f}, {disk.cy:.0f}), radius {disk.r:.0f} px")

overlay = cv2.cvtColor(example, cv2.COLOR_GRAY2RGB)
for rle in rles:
    m = mask_util.decode(rle)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, (255, 40, 40), 3)
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(prob, cmap="magma", vmin=0, vmax=1); ax[0].set_title("probability (1024 px)")
ax[1].imshow(overlay); ax[1].set_title("predicted filaments (2048 px)")
for a in ax: a.axis("off")
plt.show()
""")
md("## 5. All test images -> submission CSV")
code(r"""
started = time.time()
rows = []
for path in test_paths:
    small = downsample(read_image(path), RES)
    prob = predict(model, small[None], DEVICE)[0].astype(np.float32)
    for n, rle in enumerate(instances(prob, POST, fit_disk(small)), start=1):
        rows.append((f"{path.stem}_{n}", rle["counts"].decode()))
elapsed = time.time() - started
OUTPUT.parent.mkdir(exist_ok=True)
write_csv(rows, OUTPUT, [p.stem for p in test_paths])  # checks unique ids, known images, non-empty masks
n_images = len({r[0].rsplit('_', 1)[0] for r in rows})
print(f"{len(rows)} filaments on {n_images} of {len(test_paths)} images; "
      f"{elapsed / len(test_paths):.2f} s per image on {DEVICE}; written to {OUTPUT.relative_to(ROOT)}")
""")
md(r"""
## 6. Validation study (pre-computed)

The figures below are the organisers' three evaluation plots computed on out-of-fold predictions of
five fold models (5-fold `GroupKFold` by calendar month, 707 images, 1,154 annotator-image entries),
so every image is scored by a model that never saw it: PQ 0.409, SQ 0.666, mean IoU 0.553 and mean
Dice 0.688 over pairs with IoU > 0. They are reproduced by `python -m filseg.report` (section 7).
For reference, one annotator scored against another on the same images gives PQ 0.340.
""")
code(r"""
stats = json.loads((ROOT / "docs" / "figures" / "oof5_stats.json").read_text())
print(stats["summary"], "| mean IoU", stats["mean_iou"], "| mean Dice", stats["mean_dice"])
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for a, name in zip(ax, ["iou", "dice", "m2n"]):
    a.imshow(plt.imread(ROOT / "docs" / "figures" / f"oof5_{name}.png")); a.axis("off")
plt.show()
""")
md(r"""
## 7. Training and tuning (reference; not run here)

Run from the repository root with the project's environment (`pip install -r requirements.txt`).
Set `RUN_TRAINING = True` below to execute them from the notebook (about 1.5 h per model on an
Apple M3 Max; faster on a CUDA GPU).

| Step | Command |
|---|---|
| Cache images and consensus targets at 1024 px | `python -m filseg.cache --res 1024` |
| Train one fold (validation PQ every 5 epochs) | `python -m filseg.train --fold 0 --name unet_r34full --epochs 40` |
| Tune post-processing on pooled out-of-fold predictions | `python -m filseg.tune --run unet_r34full_f0 ... unet_r34full_f4` |
| Evaluation plots | `python -m filseg.report --run unet_r34full_f0 ... --post <post.json>` |
| Final model on all 707 images | `python -m filseg.train --fold -1 --name unet_r34full --epochs 40` |
| Submission from the command line | `python -m filseg.predict --weights weights/unet_r34_all.pt --post <post.json>` |
""")
code(r"""
RUN_TRAINING = False
if RUN_TRAINING:
    import subprocess, sys
    for args in (["filseg.cache", "--res", "1024"],
                 ["filseg.train", "--fold", "-1", "--name", "unet_r34full", "--epochs", "40"]):
        subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True)
""")
nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "pipeline.ipynb")
