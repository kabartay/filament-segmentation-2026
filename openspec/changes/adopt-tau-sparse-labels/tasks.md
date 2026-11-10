## 1. Stage 1

- [x] 1.1 PQ and sparse-label PQ' for tau 0.22-0.36 and r = 1, 0.85, 0.77, 0.65; verify: `runs/adopt-tau-sparse-labels_stage1.json`. Result: at r = 1 0.4253 / 0.4265 / 0.4267 / 0.4269 / 0.4262 / 0.4258 / 0.4238 / 0.4223 (tau 0.22 ... 0.36); at r = 0.77 (closed form) 0.3720 / 0.3736 / 0.3744 / 0.3752 / 0.3752 / 0.3756 / 0.3745 / 0.3740.
- [x] 1.2 Minimax regret over r; verify: same file, `maxregret_sim`. Result: 0.0052 / 0.0035 / 0.0024 / 0.0011 / 0.0008 / 0.0011 / 0.0032 / 0.0046; tau 0.30 is best, 0.26 costs at most 0.0024 (seed SD 0.0007).
- [x] 1.3 Decision. Result: keep tau 0.26. The minimax tau 0.30 gains at most about 0.002 under sparse labels and loses 0.0005 at r = 1; r itself is unknown and the gain is a tenth of the public-LB interval; stage 2 (refiners trained on tau-0.26 crops) was not re-run.

## 2. Stage 2

- [ ] 2.1 Not run: new crops at tau 0.30, prediction with the existing fold refiners, paired eval. Only worth it if the selection is revisited after the deadline.
