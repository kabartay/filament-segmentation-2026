## Why

PLANNED, not started. A public notebook reports that ConvNeXt-T UNet++ crop refiners added to the refiner ensemble give
+0.0022 to +0.0029 (its own validation). Our refiners (UNet++ and jitter U-Net, `ensj`, OOF PQ 0.4399) gain +0.0016
from blending two members, so a third, different backbone may add a little more. The claim is small and from another
split, so it needs our own OOF.

## What Changes

Status 2026-11-11: NOT PURSUED (modelling stopped on 2026-11-11 before the deadline, owner decision; sub8 is final).

- Train ConvNeXt-T UNet++ refiners on our rotated crops (`runs/stage2/crops.npy`, 5 folds, with the jitter set as for
  `jit`), blend by `stage2_oof.py blend`, evaluate at threshold 0.40 with `eval` and `compare`.
- Source: public notebook (read as text only). Claimed gain: ensemble +0.0022 to +0.0029.

## Non-goals

- No change to stage 1, selection or paste-back; no weight search beyond equal weights and one 1:1:1 variant.

## Falsified by

Paired gain over `ensj_pi:t0.40` below +0.001 or interval including 0, or any fold clearly negative -> reject.

## Cost

About 6-7 GPU hours (ESTIMATED), under `runs/gpu.lock`, after any running refiner training has exited.
