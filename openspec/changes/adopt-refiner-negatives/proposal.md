## Why

Stage-2 refiners are trained only on crops of seeds some annotator drew (3,827 of 4,429 base seeds; match by
IoU > DREW_IOU = 0.3) but are applied to every selected seed, so they never see a seed nobody drew and cannot
learn to shrink or erase it. The 2026-11-09 OOF error analysis (docs/DESIGN.md section 5, "Research round")
ESTIMATED +0.003 for training with undrawn seeds as all-zero targets plus a head for the share of entries that
drew the instance, and found the refiner output barely separates TP from FP (AUC 0.63).
Source: our own error analysis; no external claim.

## What Changes

- `stage2_oof.py train --negatives [--neg-weight w] [--aux-weight a]`: undrawn base seeds and undrawn jitter
  nodes (`jitter --negatives` -> `jitneg_*`) enter training with all-zero targets; optional smp classification
  head predicting drew / (entries of the image).
- `filseg.stage2`: `build_refiner(aux=)`, `predict_crops(return_aux=)`, `paste_back(drop_erased=)`; defaults
  unchanged.
- `eval --erase` (erased seeds dropped instead of kept) and `eval --drop-grid` (drop instances with a low
  predicted drew-fraction). Tag `negj`; nothing existing is overwritten.

## Non-goals

No change to stage 1, selection or the submitted pipeline unless the verdict is adopt.

## Falsified by

Paired OOF gain of negj:t0.40 over jit_pi:t0.40 (0.4383) with a 95% CI containing 0 and fewer than 4 of 5 folds
positive; same for blend upp+negj against ensj_pi:t0.40 (0.4399).

## Cost

GPU: ~3 h for 5 folds with `--extra jit --aug --epochs 15` (jit recipe is 28-36 min/fold; ~30% more crops).
CPU: jitter negatives and eval, minutes to an hour on 3 workers.
