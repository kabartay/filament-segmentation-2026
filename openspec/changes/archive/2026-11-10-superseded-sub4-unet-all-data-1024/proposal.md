# SUPERSEDED: superseded-sub4-unet-all-data-1024

## Why

SUPERSEDED: submission 4 (2026-11-04), the all-data 1024 px U-Net with the hand rule. Recorded so the reasoning
stays readable; the weights are kept in the repo for comparison.

Source: docs/DESIGN.md section 5 ("Submission 4 and the LB probe", "Notebook reproducibility", "Where PQ is
lost", "Pooled tuning, all 5 folds"), section 8, README.md.

## What Changes (what it did)

- Final stage-1 model trained on all 707 images (`runs/unet_r34full_f-1/last.pt`, 40 epochs, 140 s/epoch, final
  loss 0.3436; `weights/unet_r34_all.pt`), 1024 px, 2 views (identity + hflip), hand rule 0.45 / closing 31 px /
  area 300 / confidence 0.8. `submissions/sub4_unet_r34_alldata.csv`: 1,306 rows, 0.39 s/image.
- Numbers: 5-fold OOF PQ 0.4086 [0.399, 0.418] (TP 5025, FP 3155, FN 3174; best rule 0.4112); public LB 0.35, the
  same as submissions 2 and 3. An empty-mask diagnostic upload (owner-approved) scored LB 0.00, showing the
  empty-prediction defect was fixed.
- `pipeline.ipynb` run on CPU from the raw JPEGs wrote a CSV byte-identical to this submission.

## Replaced by

`superseded-sub5-1536-four-flips-scorer` (2026-11-05), built from `resolution-1536-four-views` (1536 px, four
flips) plus a learned component scorer instead of the hand rule. Paired OOF gain +0.0133 [0.008, 0.018]
(0.4219 vs 0.4086), better on every fold. The scorer alone on 1024 px 2 views: 0.4162 vs 0.4086, winning all
five folds (first approach change that helped).

## Non-goals

- Not the final pipeline; kept for comparison (`weights/unet_r34_all.pt`).

## Falsified by

A learned scorer or higher resolution failing to beat 0.4086 out of fold, paired. It was overtaken, as above.

## Cost

140 s/epoch on the M3 Max, 40 epochs (about 93 min); inference 0.39 s/image on MPS.
