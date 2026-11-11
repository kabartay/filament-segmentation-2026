Not pursued: modelling stopped on 2026-11-11 before the deadline (owner decision); sub8 is final. Parked for good since 2026-11-10: the 0.70 is most likely an artefact of the old defective scorer (see proposal).

## 1. Done before parking

- [x] 1.1 Read the notebook as text; verify: `grep -n "0.70" docs/DESIGN.md` -> section 6 entry (no leakage visible, magfilo paths blocked)
- [x] 1.2 Download frozen weights and match MD5 to the notebook's log; verify: `md5 <file>` -> detector dcd3053561e40ba78718546c31525d26, refiner 74991b8d95f7922f0ca6fd75f8ab4f5f (both matched)

## 2. Parked (owner decision 2026-11-10)

- [x] 2.1 Run the frozen pipeline on the 707 training images. Not done (not pursued).
- [x] 2.2 Interpret the PQ and write it to DESIGN section 6. Not done (not pursued).
- [x] 2.3 Archive as not pursued with `--skip-specs`.
