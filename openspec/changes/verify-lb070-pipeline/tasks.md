## 1. Done before parking

- [x] 1.1 Read the notebook as text; verify: `grep -n "0.70" docs/DESIGN.md` -> section 6 entry (no leakage visible, magfilo paths blocked)
- [x] 1.2 Download frozen weights and match MD5 to the notebook's log; verify: `md5 <file>` -> detector dcd3053561e40ba78718546c31525d26, refiner 74991b8d95f7922f0ca6fd75f8ab4f5f (both matched)

## 2. Parked (owner decision 2026-11-10; resume only on the owner's word)

- [ ] 2.1 Run the frozen pipeline on the 707 training images; verify: per-image masks pixel-disjoint, `scripts/experiments/bootstrap_pq.py` PQ reported
- [ ] 2.2 Interpret the PQ (see Falsified by) and write it to DESIGN section 6; verify: `grep -n "[public notebook B]" docs/DESIGN.md`
- [ ] 2.3 Archive; verify: `openspec validate verify-lb070-pipeline --strict`
