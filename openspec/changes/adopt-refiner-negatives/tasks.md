## 1. Implementation

- [ ] 1.1 Flags `train --negatives --neg-weight --aux-weight`, `jitter --negatives`, `eval --erase --drop-grid`,
  `filseg.stage2` aux head / `drop_erased`; defaults unchanged.
  Verify: smoke `FILSEG_DEVICE=cpu scripts/experiments/stage2_oof.py train --tag adopt-refiner-negatives_smoke --negatives --aux-weight 0.1 --aug --epochs 1 --folds 0 --limit 64 --batch 16`

## 2. Data and training (GPU, after PID 99590, under runs/gpu.lock)

- [ ] 2.1 `scripts/experiments/stage2_oof.py jitter --negatives` -> runs/stage2/jitneg_*; every drew == 0.
- [ ] 2.2 `train --tag negj --extra jit --aug --epochs 15 --negatives --neg-weight 0.5 --aux-weight 0.1`; log runs/stage2/negj_train.log.

## 3. Evaluation (OOF, 707 images)

- [ ] 3.1 `eval --tag negj --name pi --erase --drop-grid ...`; `compare --new negj_pi:t0.40 --old jit_pi:t0.40`.
- [ ] 3.2 `blend --tag adopt-refiner-negatives_ens --blend upp,negj`; eval; `compare` against `ensj_pi:t0.40`.
- [ ] 3.3 Drew head: AUC TP vs FP, nested-by-fold drop cut-off.

## 4. Record

- [ ] 4.1 design-entry.md with every number (the lead merges DESIGN.md).
