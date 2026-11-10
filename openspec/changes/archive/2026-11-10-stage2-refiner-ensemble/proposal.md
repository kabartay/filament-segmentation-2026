## Why

The research round of 2026-11-09 found train/test gaps in the stage-2 refiner: it trains only on selected seeds
matched by IoU > 0.3, so it never sees fragments of longer filaments or seeds cut at other thresholds, and one
architecture was used. Unlike the stage-1 ensembles, a second architecture and harder training crops were
expected to help. Landed 2026-11-10 as submission 8 (public LB 0.37 again; expected change +0.007, a quarter
of the LB interval).

Source: DESIGN.md section 5 entries "Research round (2026-11-09)", "Second refiner architecture and blend",
"Jitter refiner", "Refiner blends with the jitter model" (2026-11-09), "Submission 8 candidate (2026-11-10)".
Ideas: model diversity and refiner training as in the public notebook tushhaaarrrr "0-40" (their claimed gains
for a UNet++ member +0.002, refiner +0.006 to +0.009).

## What Changes

- Add a UNet++ (ResNet-34) refiner per fold and a "jitter" U-Net per fold trained with extra crops from tree
  nodes at other levels (matched by IoU > 0.3 or containment 0.5), per-sample flips and a small affine jitter.
- Submission 8 uses the ten refiners UNet++ + jitter (`weights/stage2_upp_f{0..4}.pt`,
  `weights/stage2_jit_f{0..4}.pt`, float16); the plain U-Nets are dropped; `load_refiners` reads the
  architecture from checkpoint keys.

## Non-goals

- A UNet++ trained on the jitter crops (no time before the deadline); refining more tree nodes than the
  chosen antichain; post-stage-2 rescoring (see `rejected-selection-variants`).

## Falsified by

A paired OOF gain over the submission-7 refiner whose interval includes zero or a fold that gets worse. At
landing: +0.0070 [0.0050, 0.0090], 5/5 folds.

## Cost

UNet++ 41-49 min per fold on MPS (loaded); jitter crops 17.5 h CPU on 3 workers (6,610 extra crops from 670
images), jitter training 28-36 min per fold; 15.9 s per image at inference on a loaded machine.
