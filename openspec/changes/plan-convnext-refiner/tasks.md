## 1. Preparation

- [ ] 1.1 Check the ConvNeXt-T encoder loads in smp with ImageNet weights; verify: `.venv/bin/python -c "import segmentation_models_pytorch as s; s.UnetPlusPlus('tu-convnext_tiny', encoder_weights=None)"`

## 2. Training

- [ ] 2.1 Train 5 folds; verify: `.venv/bin/python scripts/experiments/stage2_oof.py train --tag cnx --arch UnetPlusPlus --encoder tu-convnext_tiny --extra jit --aug` (exact flags checked first), five `runs/stage2/cnx_f*.pt`
- [ ] 2.2 Blend with `ensj`; verify: `stage2_oof.py blend --blend upp,jit,cnx --tag ens3`

## 3. Evaluation

- [ ] 3.1 `stage2_oof.py eval --tag ens3 --name pi`, then `compare --new ens3_pi:t0.40 --old ensj_pi:t0.40`; write the per-fold table here
- [ ] 3.2 Verdict against Falsified-by

## 4. Record

- [ ] 4.1 Update docs/DESIGN.md (and README, report, notebook, weights if adopted); verify: `openspec validate plan-convnext-refiner --strict`
