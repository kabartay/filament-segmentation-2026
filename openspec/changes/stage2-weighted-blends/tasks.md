## 1. Evaluation (running)

- [ ] 1.1 Blend weights w12, w13, w21, w121, w231 at thresholds 0.35/0.40/0.45 finish; verify: `ls runs/stage2/w*_pi.json` (15 PQ values, or one file per weighting with three thresholds)
- [ ] 1.2 Paired comparison of each against `ensj_pi:t0.40` with per-fold deltas; verify: `.venv/bin/python scripts/experiments/stage2_oof.py compare` (same 2,000 resamples as section 5) and write the table here
- [ ] 1.3 Decision against the Falsified-by rule (15 comparisons, 5/5 folds); verify: written verdict next to this task

## 2. If adopted

- [ ] 2.1 Change blend weights in the submission code and rebuild the CSV; verify: `.venv/bin/python scripts/make_notebook.py` and an overlap check (0 shared pixels)
- [ ] 2.2 Update notebook, README, report; verify: `grep -n "weights" README.md report/main.tex`

## 3. Record

- [ ] 3.1 Add a DESIGN section 5 entry with every number including the losers; verify: `grep -n "weighted blend" docs/DESIGN.md`
- [ ] 3.2 Archive as adopted or rejected; verify: `openspec validate stage2-weighted-blends --strict`
