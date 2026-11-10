## Why

Most remaining loss is extent: 1,169 near-miss FN (IoU 0.3-0.5) are mostly too small or too large, not
mis-grouped, and a global grow/shrink or threshold rule cannot fix per-instance extent. A per-instance
refiner can. Landed 2026-11-08 as submission 7 (public LB 0.37, submission 6: 0.36).

Source: DESIGN.md section 5 entries "Stage 2, rotated-crop refiner (2026-11-07/08)" and "Submission 7
candidate (2026-11-08)". Idea after Reche & Cid 2026 (crop segmenter with a box channel) and the public
notebook tushhaaarrrr "0-40" (rotation-normalised crop refiner; their claimed gain +0.006 to +0.009). It
differs from the axis-aligned box-crop refiner rejected on 2026-11-03 (see
`rejected-detector-and-refiner-variants`).

## What Changes

- Each instance the tree scorer selects is redrawn in a 384x128 crop along its principal axis (native
  resolution, scaled down only if needed) with the stage-1 probability, the seed and the other seeds as extra
  channels; target = mean of the matching annotators' masks.
- ResNet-34 U-Net, 30 epochs, one refiner per fold, applied to the crops of its own fold (out of fold);
  pixel threshold 0.40 fixed before the run; pasted back disjoint.
- Submission: the five fold refiners averaged, stored as float16.

## Non-goals

- An all-data refiner (the fold models are the ones evaluated); changing stage 1 or the selection.

## Falsified by

A paired OOF gain over stage 1 (0.4267) whose bootstrap interval includes zero, or a fold that gets worse.
At landing: +0.0063 [0.0033, 0.0093], no fold worse.

## Cost

Crops 1,341 s of CPU; training about 32 s/epoch on MPS, 16-20 min per fold; 6.6 s per image at inference.
