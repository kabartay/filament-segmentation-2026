## 1. Metric and validation

- [x] 1.1 `filseg/metric.py` reproduces the organisers' notebook. Verify: compare both on 15 check pairs.
  Result: MEASURED, 0.314922 from both (DESIGN.md section 1).
- [x] 1.2 Month-grouped 5-fold split. Verify: `filseg.train` folds. Result: GroupKFold by calendar month
  (same-week images would leak); 707 images.
- [x] 1.3 Annotator ceiling. Verify: `scripts/experiments/oof_analysis.py human`. Result: MEASURED, 0.3399
  (SQ 0.634) on 296 images; U-Net 0.405 on the same images (2026-11-04).

## 2. Model

- [x] 2.1 Fold 0, 40 epochs (run unet_r34full_f0). Verify: `filseg.tune`. Result: default-setting val PQ
  e5 0.331 ... e40 0.393; tuned 0.415 (threshold 0.45, closing 31, area 200, confidence 0.85), 2026-11-03.
- [x] 2.2 Five folds. Verify: `scripts/train_folds.sh 1 2 3 4`, `filseg.tune --run unet_r34full_f0 ... f4`.
  Result: pooled 5-fold OOF PQ 0.410 (best setting 0.41034) / 0.409 (fold-0 setting 0.4086).
- [x] 2.3 Final model on all 707 images. Verify: `runs/unet_r34full_f-1/last.pt` exists. Result: done
  2026-11-04 09:08, 40 epochs, 140 s/epoch, final loss 0.3436; public LB 0.35 (submission 4).

## 3. Records

- [x] 3.1 Results in DESIGN.md section 5 and README. Result: entries "Pooled tuning, all 5 folds",
  "Submission 4 and the LB probe".
