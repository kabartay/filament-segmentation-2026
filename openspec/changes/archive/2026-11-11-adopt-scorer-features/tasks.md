## 1. Evaluation
- [x] 1.1 Baseline reproduces 0.4267 at tau 0.26: `adopt_scorer_features.py score --no-nb` gave 0.4265 / 0.4267 / 0.4269 (tau 0.24 / 0.26 / 0.28).
- [x] 1.2 Capacity, station and neighbour variants, paired bootstrap per tau and per fold: runs/adopt-scorer-features_eval_nonb.json and _eval.json; none beats the baseline. Result: 15 leaves -0.0005 at tau 0.26 (95% CI -0.0028, +0.0018; 2/5 folds); 7 leaves -0.0001 (-0.0026, +0.0025); station -0.0027 (-0.0049, -0.0006; 1/5 folds); neighbours -0.0018 (-0.0046, +0.0010; 3/5 folds); combinations -0.0014 to -0.0026.
- [x] 1.3 Spread parity on 50 images: `adopt_spread_parity.py` gave ratio 1.007, no scale mismatch.
- [x] 1.4 Test has the same station codes: counted from file names, the same six in both.

## 2. Docs
- [x] 2.1 design-entry.md written and merged into docs/DESIGN.md section 5. Verdict against "Falsified by" (no variant beating the baseline at tau 0.24/0.26/0.28 on >= 4/5 folds with a paired interval above 0): FALSIFIED; all 10 variants are at or below baseline. Rejected, so no spec delta, README, report or notebook change.
