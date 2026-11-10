## 1. Round

- [x] 1.1 Kaggle notebooks since 2026-11-05. Verify: notebook reading, refs/. Result: ESTIMATED, no new technique; LB top 0.70/0.64/0.62, crowd 0.56-0.60.
- [x] 1.2 PQ-optimal selection. Verify: refit of the hurdle CV on OOF. Result: MEASURED, marginal rule y > PQ/2 = 0.213; best tau 0.28 at 0.4269 (0.26: 0.4267, 0.20: 0.4240); calibration, per-n tau, per-load tau no gain; at most +0.001.
- [x] 1.3 Stage-2 code review. Result: no hard bug; 602 of 4,429 seeds unmatched and dropped; one flip per batch; gap list in the proposal.
- [x] 1.4 OOF error analysis after stage 2. Result: SUBSAMPLE 236 images, TP 1,629, FP 769, FN 1,141; subsample PQ 0.4260 vs 0.4330 in full.
- [x] 1.5 Keep/drop rescoring. Verify: nested fold-held-out. Result: score only 0.4247, refiner only 0.4260, both 0.4275 (+0.0014); not adopted.
- [x] 1.6 Ranked plan acted on. Result: UNet++ ensemble and jitter refiner done (`stage2-refiner-ensemble`), UNet++ + jitter OOF 0.4399.

## 2. Records

- [x] 2.1 Recorded in docs/DESIGN.md section 5, "Research round (2026-11-09...)".
