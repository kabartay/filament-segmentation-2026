# SUPERSEDED: superseded-sub2-unet-fold0-40-epochs

## Why

SUPERSEDED: submission 2 (entry dated 2026-11-03), the single fold-0 U-Net trained for the full 40 epochs.
Recorded so the reasoning stays readable; it is not the current pipeline.

Source: docs/DESIGN.md section 5 ("U-Net, full 40 epochs, fold 0", "Pooled tuning, all 5 folds") and section 4.

## What Changes (what it did)

- U-Net R34 (ImageNet ResNet-34, 1 channel, soft annotator-mean target), fold 0, 40 epochs, 1024 px, flip TTA
  (identity + hflip), no refiner.
- Post-processing: threshold 0.45, closing 31 px, area >= 300, confidence 0.8 (the "fold-0 setting"; tuned best
  0.415 used area 200, confidence 0.85, but 0.88 gave 0.403 and 0.90 0.366, so 0.8 was chosen away from the cliff).
- File `submissions/sub2_unet_r34_f0e40.csv`: 1,313 filaments on 177 of 180 test images, 0.20 s/image on MPS.
- Numbers: local fold-0 val PQ 0.415 (TP 1030, FP 625, FN 625); public LB 0.35 (from 0.32); the local-to-LB gap held
  at about 0.065. Default-setting val PQ by epoch 5-40: 0.331, 0.359, 0.366, 0.372, 0.382, 0.391, 0.392, 0.393.
- The Mask R-CNN detector and the crop refiner were tested on this model and not used (refiner 0.413 at
  threshold 0.6; detector extras +0.002).

Not stated in DESIGN.md: the exact upload date of the submission (only the entry date 2026-11-03).

## Replaced by

`superseded-sub3-unet-5fold-average` (2026-11-04): average of the five fold models. Its premise was that the
fold-0 model saw only 566 of 707 images; five equal-strength models could not be measured out of fold, so it was
read from the LB.

## Non-goals

- Not evidence for later choices: one fold of 141 images.

## Falsified by

A later pipeline not beating it out of fold. Not falsified.

## Cost

Training 40 epochs on the M3 Max (per-epoch time for this fold-0 run not recorded in DESIGN.md); inference
0.20 s/image.
