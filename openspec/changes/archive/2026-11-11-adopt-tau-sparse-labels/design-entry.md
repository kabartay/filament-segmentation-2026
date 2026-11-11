- **Selection threshold under sparser test labels (2026-11-10, `scripts/experiments/adopt_tau_sparse_labels.py`,
  runs/adopt-tau-sparse-labels_stage1.json; REJECTED, keep 0.26)**: following the public notebooks' reading of the
  OOF-to-LB gap (test labels hold a fraction r of what a training annotator draws; dariushafshar fits r = 0.77), the
  cross-validated stage-1 selections at tau 0.22-0.36 were scored at r = 1, 0.85, 0.77, 0.65, in closed form and by a
  simulation dropping each ground-truth instance with probability 1 - r (20 seeds). MEASURED, stage 1, OOF on 707
  images. At r = 1: 0.4253 / 0.4265 / 0.4267 / 0.4269 / 0.4262 / 0.4258 / 0.4238 / 0.4223 for tau 0.22 ... 0.36; at
  r = 0.77: 0.3720 / 0.3736 / 0.3744 / 0.3752 / 0.3752 / 0.3756 / 0.3745 / 0.3740. Worst-case regret over r:
  0.0052 / 0.0035 / 0.0024 / 0.0011 / 0.0008 / 0.0011 / 0.0032 / 0.0046 (simulation seed SD about 0.0007). The
  minimax tau is 0.30, but its edge over 0.26 is at most about 0.002 under sparse labels and -0.0005 at r = 1, r is
  unknown, and stage 2 would have to be re-run on the new selection; kept 0.26. The direction agrees with the theory
  (sparser labels raise the break-even score), the size does not justify a change before the deadline.
