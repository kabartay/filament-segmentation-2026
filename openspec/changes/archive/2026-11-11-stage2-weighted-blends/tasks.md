## 1. Evaluation

- [x] 1.1 Blend weights w12, w13, w21, w121, w231 at thresholds 0.35/0.40/0.45. Result: all 15 variants were evaluated (docs/DESIGN.md section 5, adoption round). The per-variant files runs/stage2/w*_pi.json are not present, so the individual PQ values are not recorded here beyond the extremes in 1.2.
- [x] 1.2 Paired comparison against `ensj_pi:t0.40` (0.4399). Result: all variants at or below it; best w12 and w231 at 0.40, each -0.0001; worst w21 at 0.35, -0.0017 (95% CI -0.0031, -0.0003). Per-fold deltas and the other variants: not recorded.
- [x] 1.3 Decision against "Falsified by" (no weighted blend beats ensj with an interval excluding 0 and 5/5 folds positive, 15 comparisons): FALSIFIED, no variant is above the equal blend at all. REJECTED; sub8 keeps equal weights.

## 2. If adopted

- [x] 2.1 Change blend weights in the submission code and rebuild the CSV. Not done (not adopted).
- [x] 2.2 Update notebook, README, report. Not done (not adopted).

## 3. Record

- [x] 3.1 DESIGN section 5 entry. Result: merged (adoption round, weighted refiner blends); it lists the extremes only, not every loser.
- [x] 3.2 Archive as rejected with `--skip-specs`; the `stage2-refinement` delta is dropped.
