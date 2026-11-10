## 1. Measure

- [x] 1.1 Count 8-connected pieces per sub8 instance after paste-back; verify: pb_count.py (scratchpad) -> 329 of 4,417 instances (7.45%) have 2+ pieces (MEASURED)
- [x] 1.2 Flag `single` in `paste_back` (default off), `--single/--cap` in `stage2_oof.py eval`; verify: default `single=""` returns the pieces untouched

## 2. Evaluate at threshold 0.40 (MEASURED, 707 images, paired against `ensj_pi:t0.40` = 0.43993)

- [x] 2.1 `largest`: 0.44027, gain +0.00035 [-0.00157, +0.00236], folds [+0.0003, -0.0002, -0.0000, +0.0017, -0.0002]
- [x] 2.2 `join` (cap 40 px, small pieces beyond the cap stay separate): 0.43943, gain -0.00050 [-0.00094, -0.00013], folds all negative
- [x] 2.3 `join_drop` (cap 40 px, then largest piece): 0.43974, gain -0.00019 [-0.00081, +0.00055], folds [+0.0007, -0.0003, -0.0006, -0.0005, -0.0003]
- [x] 2.4 Verdict: not adopted (see design-entry.md); verify: `runs/stage2/adoptpb_*_compare.out`

## 3. Record

- [ ] 3.1 Lead merges design-entry.md into docs/DESIGN.md; verify: `grep -n "one piece" docs/DESIGN.md`; archive as rejected
