## Why

PLANNED, not started. Our fold models' OOF probability maps come from `best.pt`, the epoch chosen on the validation
fold, which is the same fold the OOF score is computed on. Selecting the checkpoint on the evaluation fold inflates
every OOF number (stage 1, tree, stage 2) by an unknown amount; the final models are not selected that way on test.
Honest reporting in the report needs the size of this effect.

## What Changes

Status 2026-11-11: NOT PURSUED (modelling stopped on 2026-11-11 before the deadline, owner decision; sub8 is final).

- Measurement only: produce OOF maps from `last.pt` (final epoch, no selection on the fold), rebuild the component
  tree on them, and compare OOF PQ against the `best.pt` numbers (stage 1 0.4267; stage 2 0.4399), with per-fold and
  paired bootstrap.
- If the gap is material, the report states both and DESIGN gets the correction; no pipeline change.

## Non-goals

- No retraining, no change to submitted weights, no new selection scheme.

## Falsified by

Not a hypothesis test: the outcome is a number. If the best-vs-last gap is within the bootstrap noise (about +-0.002),
the report states that selection bias is negligible.

## Cost

GPU inference of all fold models on their held-out images (< 1 GPU hour under `runs/gpu.lock`, ESTIMATED) plus a CPU
tree rebuild and evaluation.
