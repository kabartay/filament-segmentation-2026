## 1. Training (first run lost with runs/ on 2026-11-10; retrained via runs/rebuild_3_uppj.sh, log runs/rebuild_3.out)

- [x] 1.1 Five folds finish; verify: `tail runs/stage2/uppj_train.out` and `ls runs/stage2/uppj_f*.pt` (5 files). Result: 15 epochs per fold, 8,495 training crops (plain + 6,610 jitter), about 57 min per fold on MPS.
- [x] 1.2 Per-fold and pooled OOF evaluation; verify: `stage2_oof.py eval --tag uppj --thresholds 0.35,0.40,0.45 --name pi --workers 3`. Result: PQ 0.4409 / 0.4422 / 0.4413 at 0.35 / 0.40 / 0.45; SQ 0.6808, RQ 0.6496 at 0.40.

## 2. Comparison

- [x] 2.1 Blends at 0.35/0.40/0.45; verify: `runs/stage2/*_pi.json`. Result (PQ at 0.40): uppj 0.4422, uppj+jit 0.4410, uppj+upp 0.4410, upp+jit+uppj 0.4406; sub8 (ensj) 0.4399.
- [x] 2.2 Paired comparison against `ensj_pi:t0.40` (no weighted blend was adopted); verify: `stage2_oof.py compare`.

  | variant | gain vs sub8 | 95% CI | per fold (0..4) | folds > 0 |
  |---|---|---|---|---|
  | uppj | +0.0023 | [+0.0001, +0.0044] | -0.0008 -0.0003 +0.0049 +0.0044 +0.0033 | 3/5 |
  | uppj+jit | +0.0011 | [-0.0008, +0.0030] | -0.0007 -0.0011 +0.0034 +0.0025 +0.0010 | 3/5 |
  | uppj+upp | +0.0011 | [+0.0001, +0.0022] | -0.0009 +0.0010 +0.0012 +0.0016 +0.0028 | 4/5 |
  | upp+jit+uppj | +0.0007 | [-0.0006, +0.0019] | -0.0011 -0.0002 +0.0009 +0.0030 +0.0005 | 3/5 |

- [x] 2.3 Verdict against the Falsified-by rule (CI excluding 0 AND 5/5 folds positive): REJECTED. No variant has 5/5 folds positive; uppj alone has the largest gain and a CI just above 0 but loses on folds 0 and 1. sub8 stays the final submission.

## 3. If adopted

- [x] 3.1 Not adopted; no weights shipped, no new submission. The fold models stay in `runs/stage2/uppj_f*.pt` (local only).
- [x] 3.2 Not adopted; notebook, README and report unchanged.

## 4. Record

- [x] 4.1 DESIGN section 5 entry with all numbers; verify: `grep -n uppj docs/DESIGN.md`
- [x] 4.2 Archive; verify: `openspec validate stage2-uppj-refiner --strict`
