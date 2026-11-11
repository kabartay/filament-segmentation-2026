# adopt-low-tree-levels

## Why
tushhaaarrrr's public notebook builds the stage-1 component tree on 15 levels 0.05-0.80 (their text says "15 levels
0.05-0.80"; step 0.05 gives 16 values, so the exact grid is unclear) and claims "finer levels +0.0025" (their E8, vs
their own coarser tree). Our tree has 12 levels 0.25-0.80. Going 6 -> 12 levels inside 0.30-0.80 gave nothing
(DESIGN.md sections 5/6); the open question is the low end (0.10, 0.15, 0.20), which adds larger merged
candidates and faint filaments.

## What Changes

Status 2026-11-11: REJECTED (run and evaluated, not adopted; sub8 stays final).
Measurement only: build a 15-level tree (0.10-0.80, spread features), evaluate the cross-validated hurdle with
context features at tau 0.22-0.32, compare paired against runs/oof_tree12s_1536_v4.pkl (0.4267 at tau 0.26).
No pipeline change unless adopted (then a stage-1 and stage-2 crop rebuild would be needed).

## Non-goals
No stage-2 run; no change to docs/DESIGN.md, README, report, notebook.

## Falsified by
Fewer than 4/5 folds positive at the chosen tau, or a paired bootstrap interval containing 0.

## Cost
Tree build about 1-2 h on 3 workers (12-level build: 59 min on 4 workers; machine heavily loaded); eval about 15 min CPU. No GPU.
