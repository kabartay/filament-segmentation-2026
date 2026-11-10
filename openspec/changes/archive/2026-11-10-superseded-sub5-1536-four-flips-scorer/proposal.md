# SUPERSEDED: superseded-sub5-1536-four-flips-scorer

## Why

SUPERSEDED: submission 5 (2026-11-05), the all-data 1536 px U-Net with four flips and a single-threshold
component scorer. Recorded so the reasoning stays readable; the scorer weights are kept for comparison.

Source: docs/DESIGN.md section 5 ("Second-stage component scorer", "More TTA views", "Full 1536 px pipeline",
"Submission 5 candidate", "Uncertainty of the final OOF PQ"), section 3, README.md.

## What Changes (what it did)

- All-data 1536 model (`runs/unet_r34_1536_f-1/last.pt`, HF Jobs, 40 epochs, 53 s/epoch), four flips
  (identity, h, v, both), candidate components at threshold 0.45, closing 31 px, area >= 100.
- Gradient-boosted component scorer (`weights/component_scorer_1536.pkl`, fitted on all 6,182 OOF candidates),
  keep components with predicted expected PQ contribution > tau 0.20. 1,096 rows on 176 of 180 images (sub4:
  1,306), 6.4 s/image on MPS.
- Numbers: 5-fold OOF PQ 0.4219 [0.412, 0.431] (DESIGN.md section 5 also gives 0.4220 in the full-pipeline entry);
  tau curve flat over 0.15-0.25; paired gain over submission 4 +0.0133 [0.008, 0.018], no resample at or below
  zero; scorer better than the rule on every fold. Public LB 0.35 (uploaded 2026-11-05), unchanged, because the
  public interval is about +-0.027 on about 90 images.

## Replaced by

`component-tree-selection` (submission 6, 2026-11-05 LB 0.36): a tree of candidates at 12 thresholds with a
hurdle scorer and exact antichain selection, plus view spread. Paired OOF: stage 1 0.4267 [0.417, 0.436],
+0.0049 [0.0012, 0.0085] over this pipeline (0.4219). The 1536 px resolution and four flips are kept
(`resolution-1536-four-views`).

## Non-goals

- Native 2048 px: no gain over 1536 px on folds 0 and 3 (0.4035 vs 0.4029; 0.3876 vs 0.3872).

## Falsified by

A tree selector not beating the single-threshold scorer out of fold, paired. It beat it, as above.

## Cost

Training about 53 s/epoch on an HF Jobs A100 (about 36 min); inference 6.4 s/image on MPS.
