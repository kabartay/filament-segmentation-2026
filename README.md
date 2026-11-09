# Solar filament instance segmentation (MAGFiLO, Kaggle)

Entry to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026):
instance segmentation of solar filaments on GONG H-alpha full-disk images (2048 x 2048 px), scored by
Panoptic Quality (PQ) against each MAGFiLO annotator separately.

**Result.** A single U-Net (ResNet-34 encoder) at 1536 px over four flipped views, followed by learned
instance selection: candidates at twelve thresholds form a tree, a gradient-boosted hurdle model predicts each
candidate's expected PQ contribution, and the best disjoint set is chosen exactly on the tree; a second stage
redraws each chosen filament at native resolution in a crop turned along its own axis. Out-of-fold PQ 0.433
(95% bootstrap interval 0.423-0.442) on all 707 training images (5-fold split by calendar month), against 0.427
without the second stage (paired gain +0.006 [0.003, 0.009], on all five folds), 0.422 with a single threshold
and 0.409 for the 1024 px hand-rule pipeline. Public leaderboard PQ 0.37 for the full pipeline, 0.36 for the first
stage alone and 0.35 for both earlier pipelines (two decimals over about 90 images cannot resolve these gains). For scale, one human
annotator scored against another on the same images gives PQ 0.340.

## Reproduce the submitted predictions

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.12
# download the competition data and unzip it so that this path exists:
#   data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg
.venv/bin/jupyter lab pipeline.ipynb   # or: jupyter nbconvert --to notebook --execute pipeline.ipynb
```

`pipeline.ipynb` loads `weights/unet_r34_1536_all.pt`, `weights/tree_scorer_1536.pkl` and the five stage-2
refiners `weights/stage2_unet_f{0..4}.pt`, runs inference on the 180 test images and writes
`submissions/notebook_submission.csv` (byte-identical to our final submission,
`submissions/sub7_r34_1536_tree_stage2.csv`); no training is needed. It runs on CUDA, Apple MPS or CPU
(`FILSEG_DEVICE=cpu` forces CPU; about 6.6 s per image on an M3 Max, most of it the features of the tree's
candidates).

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
   nodes maximising the sum of (score - 0.26), solved exactly by dynamic programming on the forest. Out of
   fold this gains +0.005 [0.001, 0.008] over a single threshold with the same kind of scorer
   (`filseg/scorer.py`, submission 5) and +0.018 [0.013, 0.023] over the 1024 px hand rule (paired bootstrap
   over images, `scripts/experiments/bootstrap_pq.py`).
7. **Stage 2** (`filseg/stage2.py`, after the second stage of the public notebook
   [Solar Filament Segmentation 2026 (0.40)](https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-40)):
   each selected filament is redrawn in a 384 x 128 px crop turned along its principal axis, at native
   resolution, from four channels (image, stage-1 probability, the filament, the other selected filaments).
   Target: the mean mask of the annotators who drew that filament. One ResNet-34 U-Net per fold (30 epochs),
   the five averaged with four flips at test time, threshold 0.40, pasted back disjoint. Out of fold
   (each fold's crops redrawn by the refiner that did not see them) +0.006 [0.003, 0.009] over stage 1, on
   all five folds (`scripts/experiments/stage2_oof.py`).

Everything was chosen on out-of-fold predictions with the exact competition metric (`filseg/metric.py`
reproduces the organisers' self-evaluation notebook), never on the leaderboard.

## Repository layout

| Path | Contents |
|---|---|
| `pipeline.ipynb` | the whole inference pipeline, plus the validation figures and the training commands |
| `weights/unet_r34_1536_all.pt` | final model, trained on all 707 training images at 1536 px |
| `weights/tree_scorer_1536.pkl` | final tree scorer (scikit-learn hurdle models, feature names, threshold, levels) |
| `weights/stage2_unet_f{0..4}.pt` | stage-2 crop refiners, one per fold (float16 state dicts) |
| `weights/component_scorer_1536.pkl` | single-threshold component scorer (submission 5), kept for comparison |
| `weights/unet_r34_all.pt` | earlier 1024 px model (submission 4), kept for comparison |
| `filseg/` | library: data loading, cache, training, metric, post-processing, tuning, prediction, report plots |
| `scripts/` | helpers for chaining training runs, `scripts/hf/` for training on Hugging Face Jobs; `scripts/experiments/` holds the analyses |
| `docs/DESIGN.md` | design and decision log: every experiment with its numbers, including the failed ones |
| `docs/figures/` | the organisers' three evaluation plots on out-of-fold predictions, and analysis figures |
| `requirements.txt` | exact package versions used |

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
.venv/bin/python scripts/experiments/stage2_oof.py train --tag unet --epochs 30           # one refiner per fold
.venv/bin/python scripts/experiments/stage2_oof.py eval --tag unet --workers 4            # OOF gain over stage 1
.venv/bin/python scripts/experiments/stage2_oof.py export --tag unet                      # float16 weights/stage2_unet_f*.pt
.venv/bin/python -m filseg.predict --weights runs/unet_r34_1536_f-1/last.pt --views 4 --tree-scorer weights/tree_scorer_1536.pkl \
    --refiner weights/stage2_unet_f0.pt weights/stage2_unet_f1.pt weights/stage2_unet_f2.pt weights/stage2_unet_f3.pt weights/stage2_unet_f4.pt
```

One 1536 px fold takes about 40 min on an A100 or about 3 h on an M3 Max; one stage-2 refiner 16-20 min on an M3 Max.

Only the competition's training images and annotations are used. Filament spines and chirality labels
are not used, and no external filament masks are used.

## Report

`report/main.tex` is the technical report in the organisers' Overleaf template (ACM `sigconf`; `preamble.tex`
and `main.bib` come with the template), and `report/main.pdf` its local build (`cd report && tectonic -X
compile main.tex`). The figures it uses are copied into `report/figures/`, so the folder uploads to Overleaf
as it is.

## Licence

The code is released under the MIT licence (`LICENSE`). The competition data is not part of this repository.
