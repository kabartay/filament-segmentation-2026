# Solar filament instance segmentation (MAGFiLO, Kaggle 2026)

Entry to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026):
instance segmentation of solar filaments on GONG H-alpha full-disk images (2048 x 2048 px), scored by
Panoptic Quality (PQ) against each MAGFiLO annotator separately.

**Result.** A single U-Net (ResNet-34 encoder) at 1536 px, averaged over four flips, followed by a learned
component scorer that keeps only the instances expected to raise PQ. Out-of-fold PQ 0.422 on all 707 training
images (5-fold split by calendar month; mean IoU 0.573, mean Dice 0.707 over matched pairs); public
leaderboard PQ 0.35. For scale, one human annotator scored against another on the same images gives PQ 0.340.

## Reproduce the submitted predictions

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.12
# download the competition data and unzip it so that this path exists:
#   data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg
.venv/bin/jupyter lab pipeline.ipynb   # or: jupyter nbconvert --to notebook --execute pipeline.ipynb
```

`pipeline.ipynb` loads `weights/unet_r34_1536_all.pt` and `weights/component_scorer_1536.pkl`, runs inference
on the 180 test images and writes `submissions/notebook_submission.csv` (byte-identical to our final
submission); no training is needed. It runs on CUDA, Apple MPS or CPU (`FILSEG_DEVICE=cpu` forces CPU;
about 4.5 s per image on an M3 Max GPU, most of it the scorer's feature extraction).

## Method

1. **Input**: each image area-downsampled to 1536 px (better than 1024 px on four of five folds; 2048 px no better than 1536).
2. **Target**: the consensus of the annotators, i.e. the fraction marking each pixel (computed at 2048 px,
   then downsampled), so the network learns where annotators agree.
3. **Network**: `segmentation_models_pytorch` U-Net, ImageNet ResNet-34 encoder, 1 input channel; BCE + soft
   Dice; AdamW 5e-4 with warm-up and cosine decay; 40 epochs, batch 4; flips, 90-degree rotations, small
   affine, brightness and gamma augmentation.
4. **Inference**: average over four flips (identity, horizontal, vertical, both), upsampled to 2048 px.
5. **Candidates**: threshold 0.45; a 31 px closing decides which fragments belong together (the scored masks
   keep the thresholded pixels); 8-connected components of at least 100 px. Masks are disjoint by construction.
6. **Component scorer** (`filseg/scorer.py`): a gradient-boosted regressor predicts each candidate's expected
   contribution to PQ (its matched IoU per annotator entry) from 20 shape, probability, contrast and
   disk-position features, and candidates above 0.20 are kept. It was trained on the out-of-fold candidates of
   five fold models; out of fold it beats the submitted hand rule (area and mean-probability thresholds) on
   every fold, and the best pooled hand rule by 0.002 PQ.

Everything was chosen on out-of-fold predictions with the exact competition metric (`filseg/metric.py`
reproduces the organisers' self-evaluation notebook), never on the leaderboard.

## Repository layout

| Path | Contents |
|---|---|
| `pipeline.ipynb` | the whole inference pipeline, plus the validation figures and the training commands |
| `weights/unet_r34_1536_all.pt` | final model, trained on all 707 training images at 1536 px |
| `weights/component_scorer_1536.pkl` | final component scorer (scikit-learn regressor, feature names, threshold) |
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
.venv/bin/python scripts/experiments/component_features.py --name unet_r34_1536 --probs val_probs_v4.npy --out runs/oof_components_1536_v4.pkl
.venv/bin/python scripts/experiments/component_scorer.py runs/oof_components_1536_v4.pkl # cross-validated scorer
.venv/bin/python -m filseg.scorer --components runs/oof_components_1536_v4.pkl --out weights/component_scorer_1536.pkl --tau 0.20
.venv/bin/python -m filseg.train --fold -1 --name unet_r34_1536 --epochs 40              # final model on all images
.venv/bin/python -m filseg.predict --weights runs/unet_r34_1536_f-1/last.pt --views 4 --scorer weights/component_scorer_1536.pkl
```

One 1536 px fold takes about 40 min on an A100 or about 3 h on an M3 Max.

Only the competition's training images and annotations are used. Filament spines and chirality labels
are not used, and no external filament masks are used.

## Report

`report/main.tex` is the technical report in the organisers' Overleaf template (ACM `sigconf`; `preamble.tex`
and `main.bib` come with the template), and `report/main.pdf` its local build (`cd report && tectonic -X
compile main.tex`). The figures it uses are copied into `report/figures/`, so the folder uploads to Overleaf
as it is.

## Licence

The code is released under the MIT licence (`LICENSE`). The competition data is not part of this repository.
