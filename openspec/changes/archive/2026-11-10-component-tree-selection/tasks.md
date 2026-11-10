## 1. Scorer on one level (submission 5)

- [x] 1.1 Hand-rule baseline. Verify: `filseg.tune`. Result: MEASURED rule (0.45, 31, 300, 0.8) 0.4086 at 1024 px,
  2 views (2026-11-04).
- [x] 1.2 Component scorer, 20 features, tau 0.22. Verify: `scripts/experiments/component_features.py`,
  `component_scorer.py`. Result: 0.4162 (2 views) / 0.4168 (4 views) vs rule 0.4086 / 0.4117; wins on every fold.
- [x] 1.3 Full 1536 px pipeline, scorer tau 0.20. Verify: `scripts/pipeline_1536.sh`. Result: 0.4219 (0.4220 at
  tau 0.18), vs rule 0.4165; +0.0133 [0.008, 0.018] over submission 4 (0.4086), no resample at or below zero
  (`scripts/experiments/bootstrap_pq.py`). Public LB 0.35.

## 2. Tree and spread (submission 6)

- [x] 2.1 Six-level tree with context features. Verify: `scripts/experiments/component_tree.py`. Result: 36,406
  nodes; hurdle 0.4260 (tau 0.26), +0.0041 [0.0006, 0.0076] over control, 4 of 5 folds better.
- [x] 2.2 View-spread features. Verify: `scripts/experiments/oof_view_spread.py`. Result: spread_mean, spread_q90,
  spread_max, spread_rel; free at test time.
- [x] 2.3 Twelve levels 0.25-0.80, with spread. Result: ablation without spread 0.4252 (+0.0034 [-0.0005, +0.0070]);
  with spread 0.4265 / 0.4267 / 0.4269 at tau 0.24 / 0.26 / 0.28, gain +0.0046 / +0.0049 / +0.0051; per fold at
  tau 0.26: 0.434 / 0.425 / 0.411 / 0.430 / 0.434 vs 0.432 / 0.415 / 0.407 / 0.421 / 0.435. Weighted regressor
  0.4245 (worse). Adopted: hurdle + context + spread, tau 0.26.
- [x] 2.4 Submission 6. Verify: `filseg.predict --tree-scorer`. Result: OOF PQ 0.4267 [0.417, 0.436], +0.0049
  [0.0012, 0.0085] over submission 5 and +0.0182 [0.0134, 0.0233] over submission 4; SQ 0.6749, RQ 0.6322;
  1,067 filaments on 177 of 180 images, LB 0.36 (2026-11-05), 8.6 s per image; no overlapping masks.

## 3. Selection maths (2026-11-09, measured)

- [x] 3.1 tau sweep after refit: 0.20 0.4240, 0.2135 0.4250, 0.24 0.4265, 0.26 0.4267, 0.28 0.4269, 0.30 0.4262.
  The marginal rule is y > PQ/2 = 0.213; the best tau sits above because y_hat is optimistic on the selected
  antichain (mean y_hat 0.523 vs y 0.456). Selection is not where the next gain is (at most +0.001).

## 4. Records

- [x] 4.1 DESIGN.md entries as above; README and report describe the tree.
