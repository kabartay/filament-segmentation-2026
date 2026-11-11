# Tasks

- [x] 1. Smoke build, 6 images: `component_tree.py build --spread --levels ... --limit 6` -> 6 images, 402 nodes, 397 s.
- [x] 2. Full 15-level build: `component_tree.py build --spread --levels 0.10,...,0.80 --out runs/adopt-low-tree-levels_tree15s.pkl --workers 3` -> 707 images, 92,376 nodes, 15,038 s.
- [x] 3. Evaluate: `component_tree.py eval --out runs/adopt-low-tree-levels_tree15s.pkl --context --modes hurdle --taus 0.22,...,0.32` -> PQ 0.4217 at tau 0.26 (12-level 0.4267), best 0.4230 at tau 0.30; negative on 5/5 folds at tau 0.26.
- [x] 4. Paired per-fold comparison against runs/oof_tree12s_1536_v4.pkl: see design-entry.md. Verdict: reject.
- [x] 5. 14-level variant (0.15-0.80): not run, first result negative at every tau.
- [x] 6. DESIGN.md entry. Result: merged into docs/DESIGN.md section 5 (adoption round, "15-level tree"). Verdict against "Falsified by" (fewer than 4/5 folds positive, or interval containing 0): FALSIFIED. The 15-level tree is 0.4217 against 0.4267 at tau 0.26 (-0.0050 PQ), negative on 5/5 folds (fold 0-4: 0.4301/0.4184/0.4029/0.4250/0.4325 vs 0.4340/0.4247/0.4107/0.4304/0.4341); even at its best tau 0.30 (0.4230) it is 0.0032 below the 12-level best. TP 4,833 / FP 2,440 / FN 3,366 against 4,872 / 2,341 / 3,327. Against the level-0.45 control the 15-level hurdle is -0.0002 (95% CI -0.0042 to +0.0036). Rejected; 12 levels 0.25-0.80 kept.
