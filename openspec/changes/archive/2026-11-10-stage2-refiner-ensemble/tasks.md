## 1. Candidates (threshold 0.40, paired over the same 2,000 resamples, `scripts/experiments/stage2_oof.py compare`)

- [x] 1.1 UNet++ R34 refiner. Verify: `stage2_oof.py train --tag upp --arch UnetPlusPlus --epochs 30`. Result:
  MEASURED OOF at 0.35 / 0.40 / 0.45: 0.4345 / 0.4355 / 0.4351; paired over the sub-7 refiner +0.0025
  [0.0002, 0.0047] (5/5). Blend U-Net + UNet++ 0.4360 [0.4264, 0.4450], +0.0031 [0.0016, 0.0044] (5/5), over
  stage 1 +0.0093.
- [x] 1.2 Jitter refiner. Verify: `stage2_oof.py jitter`, then `train --tag jit --extra jit --aug --epochs 15`.
  Result: OOF 0.4375 / 0.4383 / 0.4381 at 0.35 / 0.40 / 0.45; +0.0054 [0.0030, 0.0080] (5/5) over the sub-7
  refiner; over the U-Net + UNet++ blend +0.0023 [0.0000, 0.0047] (4/5).
- [x] 1.3 Blend table. Result: U-Net + UNet++ + jitter 0.4391 (+0.0061 [0.0046, 0.0077]); UNet++ + jitter 0.4399
  (+0.0070 [0.0050, 0.0090], 5/5; against the jitter model alone +0.0016 [-0.0002, 0.0034], against the
  three-way blend +0.0009 [-0.0001, 0.0019]). Chosen: best on every fold among five candidates.
- [x] 1.4 Final figures for UNet++ + jitter. Result: PQ 0.4399 [0.4305, 0.4487], SQ 0.6803, RQ 0.6467; per fold
  0.4415, 0.4379, 0.4265, 0.4479, 0.4460; multi-annotator images 0.4406; +0.0132 [0.0104, 0.0160] over stage 1.

## 2. Submission 8

- [x] 2.1 `filseg.predict --tree-scorer --refiner weights/stage2_upp_f*.pt weights/stage2_jit_f*.pt`
  (`submissions/sub8_r34_1536_tree_stage2_ens.csv`). Result: 1,065 filaments on 177 of 180 images, no overlapping
  pixels; median best-match IoU against submission 7 is 0.964 (12 masks below 0.8, 1 below 0.5), area +0.2%;
  15.9 s per image (loaded machine; submission 7: 6.6 s); public LB 0.37.

## 3. Records

- [x] 3.1 DESIGN.md section 3 and section 5 updated; README, report and pipeline.ipynb describe the ten refiners.
