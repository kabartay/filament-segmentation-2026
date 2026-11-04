# Solar filament instance segmentation (MAGFiLO, Kaggle 2026)

Entry to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026):
instance segmentation of solar filaments on GONG H-alpha full-disk images (2048 x 2048 px), scored by
Panoptic Quality (PQ) against each MAGFiLO annotator separately.

**Result.** A single U-Net (ResNet-34 encoder) with tuned post-processing. Out-of-fold PQ 0.409 on all
707 training images (5-fold split by calendar month); public leaderboard PQ 0.35. For scale, one human
annotator scored against another on the same images gives PQ 0.340.

## Reproduce the submitted predictions

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.12
# download the competition data and unzip it so that this path exists:
#   data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg
.venv/bin/jupyter lab pipeline.ipynb   # or: jupyter nbconvert --to notebook --execute pipeline.ipynb
```

`pipeline.ipynb` loads `weights/unet_r34_all.pt`, runs inference on the 180 test images and writes
`submissions/notebook_submission.csv`; no training is needed. It runs on CUDA, Apple MPS or CPU
(`FILSEG_DEVICE=cpu` forces CPU; about 0.4 s per image on an M3 Max GPU).

## Method

1. **Input**: each image area-downsampled to 1024 px.
2. **Target**: the consensus of the annotators, i.e. the fraction marking each pixel (computed at 2048 px,
   then downsampled), so the network learns where annotators agree.
3. **Network**: `segmentation_models_pytorch` U-Net, ImageNet ResNet-34 encoder, 1 input channel; BCE + soft
   Dice; AdamW 5e-4 with warm-up and cosine decay; 40 epochs, batch 4; flips, 90-degree rotations, small
   affine, brightness and gamma augmentation.
4. **Inference**: average of the image and its horizontal flip, upsampled to 2048 px.
5. **Instances**: threshold 0.45; a 31 px closing decides which fragments belong together (the scored masks
   keep the thresholded pixels); 8-connected components; keep components of at least 300 px with mean
   probability at least 0.8. Masks are disjoint by construction.

The post-processing setting was tuned on the pooled out-of-fold predictions of five fold models with the
exact competition metric (`filseg/metric.py` reproduces the organisers' self-evaluation notebook).

## Repository layout

| Path | Contents |
|---|---|
| `pipeline.ipynb` | the whole inference pipeline, plus the validation figures and the training commands |
| `weights/unet_r34_all.pt` | final model, trained on all 707 training images |
| `filseg/` | library: data loading, cache, training, metric, post-processing, tuning, prediction, report plots |
| `scripts/` | helpers for chaining training runs; `scripts/experiments/` holds the one-off analyses |
| `docs/DESIGN.md` | design and decision log: every experiment with its numbers, including the failed ones |
| `docs/figures/` | the organisers' three evaluation plots on out-of-fold predictions, and analysis figures |
| `requirements.txt` | exact package versions used |

## Train from scratch

```bash
.venv/bin/python -m filseg.cache --res 1024                                   # cache images and targets
.venv/bin/python -m filseg.train --fold 0 --name unet_r34full --epochs 40     # one fold, with validation
.venv/bin/python -m filseg.tune --run unet_r34full_f0 unet_r34full_f1 unet_r34full_f2 unet_r34full_f3 unet_r34full_f4
.venv/bin/python -m filseg.train --fold -1 --name unet_r34full --epochs 40    # final model on all images
.venv/bin/python -m filseg.predict --weights runs/unet_r34full_f-1/last.pt --post <post.json>
```

Only the competition's training images and annotations are used. Filament spines and chirality labels
are not used, and no external filament masks are used.
