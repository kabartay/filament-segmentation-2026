# adopt-tau-sparse-labels: should tau move up for sparser test labels? (REJECTED, keep 0.26)

## Why
Public notebooks report an OOF-to-LB gap of 0.05-0.06 and attribute it to the labels: dariushafshar models the test
labels as holding a fraction r = 0.77 of what a training annotator draws, PQ' = r SumIoU / (0.5|P| + 0.5 r|G|), which
moves the break-even score of a candidate from PQ/2 to PQ'/(2r). tushhaaarrrr finds tau above PQ/2 because selected
nodes' scores are optimistic and picks a minimax-regret tau.

## What Changes
Nothing in the submitted pipeline. `scripts/experiments/adopt_tau_sparse_labels.py` scores the cross-validated hurdle
selections at tau 0.22-0.36 against r = 1.0, 0.85, 0.77, 0.65 (closed form, and a simulation that drops each ground-truth
instance with probability 1 - r, 20 seeds). `stage2_oof.py` gains `FILSEG_STAGE2_OUT` / `FILSEG_TAU` and a `predict`
step for re-running stage 2 on another selection (defaults unchanged; not used).

## Non-goals
Fitting r from the public LB (two decimals over about 90 images cannot resolve it).

## Falsified by
A tau whose worst-case regret over r in [0.65, 1.0] beats 0.26 by more than the seed spread of the simulation.

## Cost
CPU only, about 15 CPU-minutes (4 h wall clock under load on 2026-11-10).
