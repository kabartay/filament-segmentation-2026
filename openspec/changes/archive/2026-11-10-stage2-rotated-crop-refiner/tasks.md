## 1. Crops and training

- [x] 1.1 Build crops from the CV selections of the 5-fold OOF pipeline. Verify: `scripts/experiments/stage2_oof.py`
  (runs/stage2). Result: 4,429 crops, 3,827 drawn by at least one annotator, 1,341 s.
- [x] 1.2 Train per fold, predict each fold's crops with the model that did not see them. Result: 30 epochs, about
  32 s/epoch on MPS, 16-20 min per fold.

## 2. Evidence

- [x] 2.1 OOF pooled PQ by pixel threshold. Result: MEASURED seeds 0.4267 (reproduces the pipeline); refined 0.30
  0.4291, 0.35 0.4319, 0.40 0.4330, 0.45 0.4330, 0.50 0.4320, 0.55 0.4302. Threshold 0.40 was fixed before the
  run, not the OOF best.
- [x] 2.2 Paired gain at 0.40. Verify: `scripts/experiments/bootstrap_pq.py`. Result: +0.0063 [0.0033, 0.0093];
  per fold seed/refined 0.4340/0.4341, 0.4247/0.4326, 0.4107/0.4189, 0.4304/0.4423, 0.4341/0.4370.
- [x] 2.3 Extended eval (`--name final`). Result: PQ 0.4330 [0.4234, 0.4419], SQ 0.6771, RQ 0.6394 (stage 1:
  0.4267, 0.6749, 0.6322); gain mostly RQ; 296 multi-annotator images 0.4341.

## 3. Submission 7

- [x] 3.1 `filseg.predict --tree-scorer --refiner weights/stage2_unet_f{0..4}.pt`. Result: 1,066 filaments on 177
  of 180 images, no overlapping pixels; median best-match IoU against submission 6 is 0.911 (137 masks below
  0.8, 1 below 0.5), area +1.4%; float16 weights 49 MB each (about 1 crop pixel in 10^5 differs at 0.40);
  6.6 s per image; public LB 0.37 (2026-11-08).

## 4. Records

- [x] 4.1 DESIGN.md entries as above; README, report and notebook describe stage 2.
