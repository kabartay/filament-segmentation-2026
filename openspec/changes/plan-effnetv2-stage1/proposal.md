## Why

PLANNED, not started. dariushafshar's public notebook reports public LB 0.37 -> 0.39 after adding an EfficientNetV2-S
stage-1 member to a U-Net ensemble. Our stage 1 is a ResNet-family consensus U-Net (see `docs/DESIGN.md` section 3); a
second, architecturally different encoder is the kind of diversity that can lift the stage-1 probability map the
component tree selects from. The claim is a two-decimal LB change on ~90 images (+-0.027), so it is a lead, not evidence.

## What Changes

- Train an EfficientNetV2-S encoder U-Net on our own 5-fold GroupKFold-by-month split, 1536 px, same recipe as the
  current members; add its OOF map to the stage-1 average; rebuild the tree and re-run selection and stage 2.
- Source: dariushafshar's notebook (read as text only; never executed). Claimed gain: LB 0.37 -> 0.39 (not OOF).

## Non-goals

- No external pretrained labels or data; ImageNet encoder weights only, as for the current members.
- Not before the 2026-11-15 deadline unless the GPU is otherwise idle; no change to stage 2 or selection rules.
- The notebook's own OOF numbers are not reused: its folds split months, so its OOF claims are leak-prone.

## Falsified by

Paired OOF bootstrap (`component_tree.py eval`, then `stage2_oof.py compare`) against `oof_tree12s_1536_v4.pkl`
(stage 1 PQ 0.4267): gain interval includes 0 or any fold clearly negative -> reject.

## Cost

About 50 GPU hours on our M3 Max MPS (5 folds, one heavy job at a time) plus tree rebuild and re-evaluation on CPU.
ESTIMATED.
