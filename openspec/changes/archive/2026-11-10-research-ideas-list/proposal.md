# RESEARCH: ideas list (DESIGN.md section 8) and what became of each idea

## Why

Section 8 of DESIGN.md ranked ideas by expected value per hour, and later rounds (2026-11-05 survey,
2026-11-09 research round) added more. This change records the status of every idea at the close, 2026-11-10, and
points to the change that covers it.

## What Changes

Nothing in the pipeline; a status register.

Section 8 list:

| # | Idea | Status | Evidence / change |
|---|---|---|---|
| 1 | Re-tune limb filter with the Hough disk | done, no change | best 0.41032 same as old fit; `research-training-data-facts` |
| 2 | Limb-darkening correction as input | rejected | tuned 0.4112 vs 0.4160 (fold 0); `rejected-*` |
| 3 | Heavier encoder | rejected | ResNet-50 0.4139 vs 0.4160; averaged 0.4159; `rejected-stage1-variants` |
| 4 | Boundary-aware loss | Lovasz rejected (0.3901 vs 0.4160); a boundary/distance term never tried; one-annotator sampling judged not worth a run | `rejected-*` |
| 5 | Per-instance threshold | not run as such; global grow/shrink failed (reference 0.4112; erode 1 px 0.3986, dilate 1 px 0.3811); per-instance extent was taken by the refiner | `stage2-rotated-crop-refiner` |
| 5b | Higher input resolution | done: 1536 px adopted (mean +0.006, 4 of 5 folds); 2048 px no further gain | `resolution-1536-four-views` |
| 6 | Train on all 707 images | done 2026-11-04 (LB 0.35) | `stage1-consensus-unet` |
| 7 | Learned grouping (CondInst, DBSCAN) | open, low priority: merges and splits about a sixth of near misses; Mask R-CNN as grouping was tried on fold 0 (0.377 alone) | `research-where-pq-is-lost`, `rejected-*` |

Tried and rejected at the foot of section 8: Mask R-CNN detector, crop refiner (original axis-aligned), detector
+ U-Net combinations, ensembling with a weaker model, size-dependent confidence, hysteresis, a lower minimum area.

Ideas that arrived later:

| Idea | Source | Status |
|---|---|---|
| Component tree + hurdle scorer + antichain | public survey 2026-11-05 | done: +0.0041 (6 levels), 12 levels + spread +0.0049 [about 0.001, 0.009]; `component-tree-selection` |
| View-spread features | own | done, adopted |
| Rotated-crop stage-2 refiner | survey, Reche & Cid | done: +0.0063 [0.0033, 0.0093]; `stage2-rotated-crop-refiner` |
| Second refiner architecture (UNet++) | research round 2026-11-09 | done: blend +0.0031; `stage2-refiner-ensemble` |
| Refiner on other tree levels with jitter | research round | done: jitter +0.0054; UNet++ + jitter 0.4399, +0.0070; `stage2-refiner-ensemble` |
| 4 flips instead of 2 / 8 dihedral views | own | 4 adopted (0.4192 vs 0.4149 fold 0); 8 no gain (0.4187) |
| Scorer refit on refined-mask features / predicted-IoU head (Mask Scoring R-CNN) | literature | tested as keep/drop rescoring: +0.0014, not adopted; `research-pq-selection-and-oof-errors-2026-11-09` |
| Per-instance choice among seed and refined masks (0.30/0.40/0.50) | research round | open (est. +0.001-0.003) |
| Refiner with undrawn seeds as zero targets + agreement head | research round | open (est. +0.003) |
| Refine more tree nodes than the chosen antichain | research round | open (est. +0.004, about 3x crops) |
| UNet++ trained on the jitter crops | own | not tried, no time before the deadline |
| Ordinal agreement target, boundary-band loss, RankSEG extents | literature | open, low priority |
| ConvNeXt-T member | survey (+0.003) | fold 0 +0.0016 (0.4261 vs 0.4245), not continued: about 50 GPU hours |
| 15 tree levels, RF-DETR scorer features, weight EMA, one connected piece per instance | LB-group notebooks | open (claimed +0.0025, +0.0034, n/a, n/a) |
| Per-annotator heads | identity codes in training ids only | not pursued; test set has no codes |
| Kaggle GPU / Kaggle CPU as training host | own | not used: GPU sessions taken (2026-11-07); CPU 25-40x slower than the M3 Max (2026-11-10) |
| HF Jobs A100 for training | own | used (folds 3, 4, all-data 1536, 2048 px folds 0 and 3, view spread) |

## Non-goals

No new experiments; no GPU or CPU jobs.

## Falsified by

n/a (a register); each entry is falsified or confirmed by the change cited.

## Cost

None.
