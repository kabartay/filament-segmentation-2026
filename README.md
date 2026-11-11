# Solar filament instance segmentation (MAGFiLO, Kaggle)

Entry to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026):
instance segmentation of solar filaments on GONG H-alpha full-disk images (2048 x 2048 px), scored by
Panoptic Quality (PQ) against each MAGFiLO annotator separately.

**Result.** One U-Net (ResNet-34 encoder) at 1536 px over four flipped views; learned instance selection on a
tree of candidates at twelve thresholds; a second stage that redraws each selected filament at native resolution in
a crop turned along its own axis. Out-of-fold PQ **0.440** (95% bootstrap interval 0.431-0.449) on all 707 training
images (5-fold split by calendar month), public leaderboard 0.37. For scale, one human annotator scored against
another on the same images gives PQ 0.340.

| Submission | Pipeline | Local PQ | Public LB |
|---|---|---|---|
| 1 | U-Net R34 1024 px, fold 0, epoch 10, hand rule | 0.389 (fold 0) | 0.32 |
| 2 | the same, 40 epochs | 0.415 (fold 0) | 0.35 |
| 3 | average of 5 fold models, hand rule | 0.410 (OOF) | 0.35 |
| 4 | one model on all images, hand rule | 0.409 (OOF) | 0.35 |
| 5 | 1536 px, 4 flips, single threshold + learned scorer | 0.422 (OOF) | 0.35 |
| 6 | 12-level tree + hurdle scorer + exact selection (stage 1) | 0.427 (OOF) | 0.36 |
| 7 | stage 1 + one U-Net crop refiner per fold | 0.433 (OOF) | 0.37 |
| **8 (final)** | stage 1 + UNet++ and jitter U-Net refiners, ten models | **0.440 (OOF)** | 0.37 |

From submission 4 on, the submitted model is trained on all images and the local PQ is that of the same pipeline
run on the five fold models' out-of-fold predictions. Every step from 5 to 8 was adopted on a paired bootstrap over
images of those predictions; submission 8
gains +0.013 [0.010, 0.016] over stage 1 and +0.007 [0.005, 0.009] over submission 7, on all five folds. The public
leaderboard scores about 90 images to two decimals (about +-0.027), so it cannot resolve these steps and was never
used to choose. `filseg/metric.py` reproduces the organisers' PQ notebook exactly.

## Reproduce the submitted predictions

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.12
# download the competition data and unzip it so that this path exists:
#   data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg
.venv/bin/jupyter lab pipeline.ipynb   # or: jupyter nbconvert --to notebook --execute pipeline.ipynb

.venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m pytest   # optional: 45 tests, about 10 s
```

`pipeline.ipynb` loads `weights/unet_r34_1536_all.pt`, `weights/tree_scorer_1536.pkl` and the ten stage-2
refiners `weights/stage2_upp_f{0..4}.pt` and `weights/stage2_jit_f{0..4}.pt`, runs inference on the 180 test images and writes
`submissions/notebook_submission.csv` (byte-identical to our final submission,
`submissions/sub8_r34_1536_tree_stage2_ens.csv`); no training is needed. It runs on CUDA, Apple MPS or CPU
(`FILSEG_DEVICE=cpu` forces CPU; about 16 s per image on a shared M3 Max, most of it the ten stage-2
refiners; 6.6 s with the five refiners of submission 7).

## Method

1. **Input**: each image area-downsampled to 1536 px (better than 1024 px on four of five folds; 2048 px no better than 1536).
2. **Target**: the consensus of the annotators, i.e. the fraction marking each pixel (computed at 2048 px,
   then downsampled), so the network learns where annotators agree.
3. **Network**: `segmentation_models_pytorch` U-Net, ImageNet ResNet-34 encoder, 1 input channel; BCE + soft
   Dice; AdamW 5e-4 with warm-up and cosine decay; 40 epochs, batch 4; flips, 90-degree rotations, small
   affine, brightness and gamma augmentation.
4. **Inference**: four flipped views (identity, horizontal, vertical, both); their mean is the probability map
   and their per-pixel standard deviation the view spread (`filseg.train.predict_with_spread`); both upsampled
   to 2048 px.
5. **Candidate tree** (`filseg/tree.py`): at each of twelve thresholds (0.25-0.80) a 31 px closing decides which
   fragments belong together (the masks keep the thresholded pixels) and 8-connected components of at least
   100 px are candidates. A component at a higher threshold lies inside one at the next lower threshold, so
   the candidates form a forest.
6. **Selection**: a hurdle model, P(match) x E[matched IoU per annotator entry | match], both gradient-boosted
   and trained on out-of-fold nodes of five fold models, scores every node from 32 features (shape,
   probability, contrast, disk position, tree context, view spread). The submission is the disjoint set of
   nodes maximising the sum of (score - 0.26), solved exactly by dynamic programming on the forest. The threshold
   tree, the hurdle target and the exact selection follow the first stage of the public notebook
   [Solar Filament Segmentation 2026 (0.40)](https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-40);
   the implementation and the features (including the view spread) are ours. Out of
   fold this gains +0.005 [0.001, 0.008] over a single threshold with the same kind of scorer
   (`filseg/scorer.py`, submission 5) and +0.018 [0.013, 0.023] over the 1024 px hand rule (paired bootstrap
   over images, `scripts/experiments/bootstrap_pq.py`).
7. **Stage 2** (`filseg/stage2.py`, after the second stage of the same public notebook):
   each selected filament is redrawn in a 384 x 128 px crop turned along its principal axis, at native
   resolution, from four channels (image, stage-1 probability, the filament, the other selected filaments).
   Target: the mean mask of the annotators who drew that filament. Two refiner kinds, one model per fold
   each, ResNet-34 encoder: a UNet++ trained 30 epochs on these crops, and a U-Net trained 15 epochs on them
   plus crops of tree nodes at other thresholds (fragments and over-merged seeds, matched to an annotator by
   IoU > 0.3 or by lying half inside one) with per-sample flips and a small affine jitter. All ten averaged
   with four flips at test time, threshold 0.40, pasted back disjoint. Out of fold (each fold's crops redrawn
   by the refiners that did not see them) +0.013 [0.010, 0.016] over stage 1, on all five folds; +0.007
   [0.005, 0.009] over the single U-Net refiner of submission 7 (`scripts/experiments/stage2_oof.py`).

**What did not help** (each measured out of fold, numbers in `docs/DESIGN.md`): a Mask R-CNN detector and
detector + U-Net combinations; an axis-aligned crop refiner; ResNet-50 and ConvNeXt-T encoders; limb-darkening
correction; Lovasz loss; 2048 px input; post-processing variants (hysteresis, size-dependent confidence,
growing or shrinking masks); 15 tree levels; more scorer capacity and extra features; a different tau; weighted refiner
blends; a refiner trained with negative crops; making each instance one piece; a UNet++ on the jitter crops
(+0.002, not on every fold).

## Repository layout

| Path | Contents |
|---|---|
| `pipeline.ipynb` | the whole inference pipeline, plus the validation figures and the training commands (generated by `scripts/make_notebook.py`) |
| `weights/unet_r34_1536_all.pt` | stage-1 U-Net, trained on all 707 training images at 1536 px (93 MB) |
| `weights/tree_scorer_1536.pkl` | tree scorer: scikit-learn hurdle models, feature names, tau and levels (3 MB) |
| `weights/stage2_upp_f{0..4}.pt`, `weights/stage2_jit_f{0..4}.pt` | stage-2 crop refiners, UNet++ and U-Net with extra crops, one per fold (float16 state dicts, 47-50 MB each) |
| `weights/component_scorer_1536.pkl` | single-threshold scorer of submission 5, kept for comparison |
| `filseg/` | library; the submitted path is `cache`, `data`, `train`, `geometry`, `postprocess`, `tree`, `stage2`, `predict`, `metric`; modules marked "Status" in their docstring are earlier experiments |
| `scripts/` | training helpers and host launchers, described in `scripts/README.md`; `scripts/experiments/` holds the out-of-fold analyses (`scripts/experiments/README.md`) |
| `tests/` | 45 fast tests (metric, tree selection, stage-2 paste-back, geometry, weight loading); no competition data needed |
| `docs/DESIGN.md` | design and decision log: every experiment with its numbers, including the failed ones |
| `docs/figures/` | the organisers' three evaluation plots on out-of-fold predictions, and analysis figures |
| `openspec/` | specifications of the pipeline (`openspec/specs/`) and every change proposed, adopted or rejected, with its falsification rule (`openspec/changes/archive/`) |
| `report/` | the 4-page technical report (LaTeX source and PDF) |
| `requirements.txt`, `requirements-dev.txt` | exact package versions used; the dev file adds pytest |

The two `.pkl` files are pickled scikit-learn models: they load with the pinned scikit-learn 1.9.1, and like any
pickle they run code when loaded, so load them only from this repository.

## Train from scratch

```bash
export PYTHONPATH=. FILSEG_CACHE=data/cache1536
.venv/bin/python -m filseg.cache --res 1536                                              # cache images and targets
for k in 0 1 2 3 4; do .venv/bin/python -m filseg.train --fold $k --name unet_r34_1536 --epochs 40; done
.venv/bin/python scripts/experiments/oof_probs_views.py --views 4 --name unet_r34_1536   # OOF probabilities, 4 flips
.venv/bin/python scripts/experiments/oof_view_spread.py --name unet_r34_1536             # OOF spread of the 4 views
L=0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8
.venv/bin/python scripts/experiments/component_tree.py build --spread --levels $L --out runs/oof_tree12s_1536_v4.pkl
.venv/bin/python scripts/experiments/component_tree.py eval --out runs/oof_tree12s_1536_v4.pkl --context --modes hurdle  # cross-validated, choice of tau
.venv/bin/python -m filseg.tree --nodes runs/oof_tree12s_1536_v4.pkl --out weights/tree_scorer_1536.pkl --tau 0.26
.venv/bin/python -m filseg.train --fold -1 --name unet_r34_1536 --epochs 40              # final model on all images
.venv/bin/python scripts/experiments/stage2_oof.py crops --workers 4                    # stage-2 crops from OOF selections
.venv/bin/python scripts/experiments/stage2_oof.py train --tag upp --arch UnetPlusPlus --epochs 30  # UNet++ refiner per fold
.venv/bin/python scripts/experiments/stage2_oof.py jitter --workers 4                   # extra crops from other tree levels
.venv/bin/python scripts/experiments/stage2_oof.py train --tag jit --extra jit --aug --epochs 15     # U-Net refiner per fold
.venv/bin/python scripts/experiments/stage2_oof.py blend --tag ensj --blend upp,jit       # average of the two
.venv/bin/python scripts/experiments/stage2_oof.py eval --tag ensj --workers 4            # OOF gain over stage 1
.venv/bin/python scripts/experiments/stage2_oof.py export --tag upp                       # float16 weights/stage2_upp_f*.pt
.venv/bin/python scripts/experiments/stage2_oof.py export --tag jit                       # float16 weights/stage2_jit_f*.pt
.venv/bin/python -m filseg.predict --weights runs/unet_r34_1536_f-1/last.pt --views 4 --tree-scorer weights/tree_scorer_1536.pkl \
    --refiner weights/stage2_upp_f*.pt weights/stage2_jit_f*.pt
```

One 1536 px fold takes about 40 min on an A100 or about 3 h on an M3 Max; one stage-2 refiner fold 16-20 min (U-Net), 41-49 min (UNet++) or 28-36 min (U-Net with
the extra crops) on an M3 Max; building the stage-2 crops and the extra crops about 17 min each on CPU (4 workers).

Only the competition's training images and annotations are used. Filament spines and chirality labels
are not used, and no external filament masks are used.

## Report

`report/main.tex` is the technical report in the organisers' Overleaf template (ACM `sigconf`; `preamble.tex`
and `main.bib` come with the template), and `report/main.pdf` its local build (`cd report && tectonic -X
compile main.tex`). The figures it uses are copied into `report/figures/`, so the folder uploads to Overleaf
as it is.

## Data and credits

The images and annotations are those of the Kaggle *Solar Filament Segmentation Challenge 2026*
(A. Ahmadzadeh, D. J. Kempton, Q. Li and A. A. Pevtsov, 2026), built from the MAGFiLO dataset
(Ahmadzadeh et al. 2024, *A dataset of manually annotated filaments from H-alpha observations*, Scientific Data 11,
1031, https://doi.org/10.1038/s41597-024-03876-y). They are not part of this repository; download them from the
competition page. The weights in `weights/` were trained on the competition's training set, and the figures in
`docs/figures/` and `report/figures/` show crops of its images. The competition data may be used for non-commercial
purposes only (competition rules, section 4), so the weights and figures are provided for the competition's evaluation,
research and education; the MIT licence covers the code. The images are NSO/GONG data (https://gong.nso.edu/).

This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, managed by the National Solar
Observatory, which is operated by the Association of Universities for Research in Astronomy (AURA), Inc. under a
cooperative agreement with the National Science Foundation and with contribution from the National Oceanic and
Atmospheric Administration. The GONG network of instruments is hosted by the Big Bear Solar Observatory, High
Altitude Observatory, Learmonth Solar Observatory, Udaipur Solar Observatory, Instituto de Astrofísica de Canarias,
and Cerro Tololo Interamerican Observatory.

The threshold tree with its hurdle scorer and exact selection, and the stage-2 refiner, follow the public notebook
[Solar Filament Segmentation 2026 (0.40)](https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-40).

Earlier submissions' weights (the submission-7 U-Net refiners and the submission-4 1024 px model) are not
distributed; the submitted pipeline (submission 8) needs only the files listed above.

## Licence

The code is released under the MIT licence (`LICENSE`). The competition data is not part of this repository.
