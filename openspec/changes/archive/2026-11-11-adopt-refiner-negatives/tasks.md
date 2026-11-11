## 1. Implementation

- [x] 1.1 Flags `train --negatives --neg-weight --aux-weight`, `eval --erase --drop-grid`, `filseg.stage2` aux head / `drop_erased`; defaults unchanged. Result: implemented and used for the negj run and evaluation (docs/DESIGN.md section 5, "refiners trained with undrawn seeds"). The smoke-run outcome is not recorded. `jitter --negatives` was not run (see 2.1).

## 2. Data and training (GPU)

- [x] 2.1 `jitter --negatives` -> runs/stage2/jitneg_*. Not done: jitter extraction costs about 17.5 h CPU and was not rebuilt, so every jit crop is a drawn node. Only the 602 undrawn base seeds were added as all-zero targets.
- [x] 2.2 `train --tag negj --extra jit --aug --epochs 15 --negatives --neg-weight 0.5 --aux-weight 0.1`. Result: trained 06:45 to 09:05 on 2026-11-10, about 28 min per fold. Log path runs/stage2/negj_train.log is not recorded and no negj files are present under runs/stage2/ (not recorded).

## 3. Evaluation (OOF, 707 images, paired bootstrap, 2000 resamples)

- [x] 3.1 negj alone at t0.40 against jit_pi:t0.40. Result: PQ 0.4405 against 0.4383, +0.0022 (95% CI +0.0004, +0.0040), per fold +0.0040, +0.0007, +0.0022, +0.0021, +0.0020 (5 of 5 positive); the ESTIMATED +0.003 was not reached. Erasing (`--erase`) changes nothing (identical PQ): the refiner never empties a seed. Dropping by predicted drew fraction (negj, t0.40) gives PQ 0.4408 at cut-off 0.65, 0.4384 at 0.75, 0.4352 at 0.80, 0.4259 at 0.85, 0.4069 at 0.90; 0.05 to 0.30 drop nothing (0.4405).
- [x] 3.2 Blend upp+negj against `ensj_pi:t0.40` (0.4399). Result: PQ 0.4404, +0.0005 (CI -0.0005, +0.0015), per fold +0.0007, 0.0000, -0.0009, +0.0011, +0.0015 (3 of 5 positive).
- [x] 3.3 Drew head. Result: AUC TP vs FP stage-1 seeds 0.730 (per fold 0.714 to 0.751; 4,429 seeds, 75.3% TP), against 0.598 for the refiner's own peak probability here and 0.63 in the earlier analysis; correlation with the true drew fraction 0.36. Nested-by-fold drop cut-off: 0.65 in 3 folds, no drop in 2; PQ 0.4402, gain -0.0003 (CI -0.0013, +0.0006).

## 4. Record

- [x] 4.1 design-entry.md and DESIGN entry. Result: merged into docs/DESIGN.md section 5. Verdict against "Falsified by" (CI containing 0 and fewer than 4 of 5 folds positive, for negj vs jit_pi and for upp+negj vs ensj_pi): the blend is FALSIFIED (CI contains 0, 3 of 5 folds); negj alone as a single refiner is not falsified (+0.0022, CI excludes 0, 5 of 5). Rejected for the submission: no gain over the submitted ensemble, drop filter no gain. Negatives are worth keeping if the ensemble is ever retrained (not tested with jitter negatives).
