# Tasks

- [x] 1. Smoke build, 6 images: `component_tree.py build --spread --levels ... --limit 6` -> 6 images, 402 nodes, 397 s.
- [x] 2. Full 15-level build: `component_tree.py build --spread --levels 0.10,...,0.80 --out runs/adopt-low-tree-levels_tree15s.pkl --workers 3` -> 707 images, 92,376 nodes, 15,038 s.
- [x] 3. Evaluate: `component_tree.py eval --out runs/adopt-low-tree-levels_tree15s.pkl --context --modes hurdle --taus 0.22,...,0.32` -> PQ 0.4217 at tau 0.26 (12-level 0.4267), best 0.4230 at tau 0.30; negative on 5/5 folds at tau 0.26.
- [x] 4. Paired per-fold comparison against runs/oof_tree12s_1536_v4.pkl: see design-entry.md. Verdict: reject.
- [x] 5. 14-level variant (0.15-0.80): not run, first result negative at every tau.
- [ ] 6. DESIGN.md entry: lead merges design-entry.md (not edited here).
