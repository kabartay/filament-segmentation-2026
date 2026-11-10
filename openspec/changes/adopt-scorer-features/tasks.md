## 1. Evaluation
- [x] 1.1 Baseline reproduces 0.4267 at tau 0.26: `adopt_scorer_features.py score --no-nb` gave 0.4265 / 0.4267 / 0.4269.
- [x] 1.2 Capacity, station and neighbour variants, paired bootstrap per tau and per fold: runs/adopt-scorer-features_eval_nonb.json and _eval.json; none beats the baseline.
- [x] 1.3 Spread parity on 50 images: `adopt_spread_parity.py` gave ratio 1.007, no scale mismatch.
- [x] 1.4 Test has the same station codes: counted from file names, the same six in both.

## 2. Docs
- [x] 2.1 design-entry.md written. Rejected, so no spec delta, README, report or notebook change.
