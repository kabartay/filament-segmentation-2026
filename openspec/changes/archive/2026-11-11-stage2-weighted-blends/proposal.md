## Why

Submission 8 averages the UNet++ and jitter U-Net refiners with equal weights (`ensj`, 0.4399 OOF at threshold 0.40).
DESIGN section 5 (refiner blends with the jitter model) shows the jitter model alone at 0.4383 and the equal blend
0.0016 above it, so unequal weights might be worth a little more. The choice among blends was called "barely
matters"; this change measures it instead of assuming it. Only the blend weights vary; no new training.

## What Changes

Status 2026-11-11: REJECTED (run and evaluated, not adopted; sub8 stays final).

- Weighted averages of the stored OOF refiner probabilities (`stage2_oof.py blend --weights`): w12 (UNet++ : jitter
  = 1:2), w13, w21, w121 (UNet++ : jitter : plain U-Net = 1:2:1), w231, each at thresholds 0.35 / 0.40 / 0.45.
- Paired bootstrap of each against `ensj_pi:t0.40` (the submission 8 setting), with per-fold results.
- Adopted only if a candidate beats `ensj` with an interval excluding 0 and wins 5/5 folds; then the weights are
  changed in `load_refiners`/the notebook and a new submission is built (otherwise sub8 stays).
- Status 2026-11-10: the blend evaluation is RUNNING; results land in `runs/stage2/w*_pi.json`.
- Capability note: the delta is written as `## ADDED Requirements` under `stage2-refinement`, which does not exist
  in `openspec/specs/` yet (another agent is creating it by archiving). It only takes effect if a weighted blend is adopted;
  if rejected, the change is archived as rejected and the delta dropped.

## Non-goals

- No new refiner training (see `stage2-uppj-refiner`), no change to stage 1, the tree or the paste-back rules.
- No weight search finer than the five listed grids (the grid is already a multiple-comparisons risk).
- The public leaderboard is not evidence.

## Falsified by

No weighted blend beats `ensj_pi:t0.40` by a paired bootstrap interval that excludes 0 with 5/5 folds positive, after
noting that five weightings x three thresholds are 15 comparisons (a best-of-15 gain below about +0.001 is treated
as noise). Then sub8 keeps equal weights.

## Cost

CPU only, blending stored probabilities and evaluating: about 15 evaluations; no GPU. Runs next to the training job
without touching the GPU.
