## 1. Resolution

- [x] 1.1 Folds 0-4 at 1536 px. Verify: `scripts/run_1536.sh`, `NAME=unet_r34_1536 ... scripts/train_folds.sh 1 2 3 4`,
  `filseg.tune` per run. Result: MEASURED tuned 1536 vs 1024 per fold +0.009, -0.002, +0.009, +0.005, +0.007
  (mean +0.006, better on four of five); fold 0 0.4245 vs 0.4153. Adopted.
- [x] 1.2 2048 px native, folds 0 and 3 (HF Jobs A100, 76 s/epoch, about $2.75 and 66 min per fold). Result:
  default-setting best val PQ f0 0.4035 (1536: 0.4029), f3 0.3876 (1536: 0.3872); no gain, no 2048 all-data model.
- [x] 1.3 All-data 1536 model. Verify: `runs/unet_r34_1536_f-1/last.pt`. Result: HF Jobs, 40 epochs, 53 s/epoch.

## 2. Views

- [x] 2.1 Views on fold 0. Verify: `scripts/experiments/tta_views.py`. Result: identity+hflip 0.4149; 4 flips
  0.4192; 8 dihedral views 0.4187 (re-tuned 0.4193), so 4 flips.
- [x] 2.2 View-spread maps for all OOF images. Verify: `scripts/experiments/oof_view_spread.py`. Result:
  mean of four views matches `val_probs_v4.npy` to 0.0002; MPS vs CUDA spread max per-pixel difference 0.008.
- [x] 2.3 Folds 3 and 4 trained on CUDA re-predicted on MPS. Result: probabilities differ by up to 0.015
  (device rounding only).

## 3. Records

- [x] 3.1 DESIGN.md entries as above; README states 1536 px and four views.
