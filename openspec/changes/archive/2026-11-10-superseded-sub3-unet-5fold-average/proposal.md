# SUPERSEDED: superseded-sub3-unet-5fold-average

## Why

SUPERSEDED: submission 3 (2026-11-04), the average of the five cross-validation fold models. Recorded so the
reasoning stays readable; it is not the current pipeline.

Source: docs/DESIGN.md section 5 ("Pooled tuning, all 5 folds", "Two-model average, fold 0", "Submission 3")
and section 4 step 5b.

## What Changes (what it did)

- Average of the five fold U-Nets (R34, 40 epochs, 1024 px), flip TTA, setting (0.45, 31 px closing, area 300,
  confidence 0.8, limb 90 = off). CSV `submissions/sub3_unet_r34_5fold.csv`: 1,224 rows, 1.17 s/image.
- Numbers: pooled 5-fold OOF PQ 0.410 (best setting 0.41034 at 0.45/31/200/0.85/limb 70) / 0.409 (fold-0
  setting, 0.4086); default-setting val PQ f0 0.393, f1 0.389, f2 0.375, f3 0.376, f4 0.400; public LB 0.35, the
  same as the single fold-0 model. The fold-0 setting was kept because averaged probabilities are lower and
  confidence 0.8 beat 0.85 in the two-model test.
- Caveat recorded then: averaging a 40-epoch and an epoch-20 model (0.4160 and 0.3954) gave 0.4099, between the
  two; the gain of five equal-strength models could not be measured out of fold.

## Replaced by

`superseded-sub4-unet-all-data-1024` (2026-11-04): one model trained on all 707 images instead of an average of
models each missing a fold. DESIGN.md records that the ensemble "brings no visible LB gain at two decimals" and
that remaining time went to the final model and the deliverables; it does not give a longer rationale.

## Non-goals

- Not a claim that ensembling never helps: only that this one was invisible on the two-decimal LB.

## Falsified by

Not applicable (superseded without being falsified); the out-of-fold figure of the fold models (0.409) is the
reference for sub4's rule setting (0.4086).

## Cost

Five 40-epoch folds on the M3 Max (`scripts/train_folds.sh 1 2 3 4`); inference 1.17 s/image (about 6x the
single model).
