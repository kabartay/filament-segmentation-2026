## 1. Training (first run lost with runs/ on 2026-11-10; retraining via runs/rebuild_3_uppj.sh, log runs/rebuild_3.out)

- [ ] 1.1 Five folds finish; verify: `tail runs/stage2/uppj_train.out` and `ls runs/stage2/uppj_f*.pt` (5 files)
- [ ] 1.2 Per-fold and pooled OOF evaluation; verify: `PYTHONPATH=. FILSEG_CACHE=data/cache1536 .venv/bin/python scripts/experiments/stage2_oof.py eval --tag uppj --thresholds 0.35,0.40,0.45 --name pi --workers 3` writes `runs/stage2/uppj_pi.json` and `uppj_pi_per_image.npz`

## 2. Comparison

- [ ] 2.1 Blends uppj, uppj+jit, uppj+upp+jit at 0.40 (and 0.35/0.45); verify: `stage2_oof.py blend` outputs `runs/stage2/*_pi.json`
- [ ] 2.2 Paired comparison against `ensj_pi:t0.40` and any weighted blend; verify: `stage2_oof.py compare`, table with per-fold deltas written here
- [ ] 2.3 Verdict against the Falsified-by rule; verify: written next to this task

## 3. If adopted

- [ ] 3.1 Convert weights to float16 and rebuild the submission; verify: `ls -la weights/stage2_uppj_f*.pt` and overlap check (0 shared pixels)
- [ ] 3.2 Update notebook, README, report; verify: `.venv/bin/python scripts/make_notebook.py`, `grep -n uppj README.md report/main.tex`

## 4. Record

- [ ] 4.1 DESIGN section 5 entry with all numbers; verify: `grep -n uppj docs/DESIGN.md`
- [ ] 4.2 Archive; verify: `openspec validate stage2-uppj-refiner --strict`
