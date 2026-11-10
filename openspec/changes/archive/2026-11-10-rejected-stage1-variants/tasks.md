## 1. Measured results (fold 0 unless stated; tuned PQ)

- [x] 1.1 ResNet-50 encoder (`runs/unet_r50_f0`, 2026-11-04). Result: MEASURED tuned 0.4139 vs ResNet-34 0.4160;
  default-setting best 0.3934 vs 0.393. No gain.
- [x] 1.2 ResNet-34 + ResNet-50 averaged. Result: 0.4159 (SQ 0.6693 vs 0.6680, TP falls). No gain: the encoders
  make the same mistakes, which come from the annotators.
- [x] 1.3 Averaging with a weaker (epoch-20) model. Result: 0.4099, between the two (0.4160 and 0.3954).
- [x] 1.4 Limb-darkening correction as input (`filseg/flatten.py`, `runs/unet_r34flat_f0`). Result: tuned 0.4112
  (threshold 0.65) vs plain 0.4160; averaged with plain 0.4140 at threshold 0.55. No gain.
- [x] 1.5 Lovasz hinge loss (`--loss bce_lovasz`, `runs/unet_r34lovasz_f0`). Result: tuned 0.3901 vs 0.4160;
  SQ 0.651-0.656 vs 0.668. Failed.
- [x] 1.6 Five-fold model average vs a single model (submissions 3 and 2). Result: public LB 0.35 for both
  (1.17 s/image vs 0.20); the all-data single model (submission 4) also 0.35. Single model kept.
- [x] 1.7 ConvNeXt-T U-Net at 1536 px (`unet_cnxt_1536_f0`, 2026-11-06/07). Result: tuned 0.4161 vs ResNet-34
  0.4245; average of the two 0.4261, +0.0016 over ResNet-34, below fold resolution; interrupted at epoch 31
  (best.pt epoch 30); folds 1-4 not run.
- [x] 1.8 Native 2048 px input, folds 0 and 3 (2026-11-05). Result: default-setting best 0.4035 vs 0.4029 (f0) and
  0.3876 vs 0.3872 (f3). No gain; no 2048 px all-data model.
- [x] 1.9 Limb filter re-tuned with the Hough disk (5-fold OOF, 2026-11-04). Result: best 0.41032 vs 0.41034,
  the 70-degree cut changes no TP; filter stays off (90).

## 2. Records

- [x] 2.1 All results are in DESIGN.md section 5; the "Where the ceiling is" entry concludes that four model
  changes moved fold-0 PQ by -0.026 to +0.000.
