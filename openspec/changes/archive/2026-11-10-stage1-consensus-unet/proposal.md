## Why

Annotators disagree: inter-annotator PQ is 0.3399 (SQ 0.634) on the 296 training images with 2-3 annotators
(DESIGN.md section 2, 2026-11-04). Training on one annotator copies that noise, so the model is trained on
the consensus and validated out of fold. Landed 2026-11-04 (all-data model finished 09:08).

Source: DESIGN.md sections 2, 3 (items 1-4), 5 ("U-Net, full 40 epochs, fold 0", "Annotator ceiling",
"Pooled tuning, all 5 folds", "Submission 4", "Notebook reproducibility").

## What Changes

- Stage 1: `smp.Unet`, ImageNet ResNet-34, 1 input channel, target = fraction of annotators marking each pixel
  (soft annotator mean), BCE + soft Dice, AdamW 5e-4 with warm-up and cosine decay, batch 4, 40 epochs,
  flips, 90-degree rotations, +-30 deg affine, brightness/gamma.
- 5-fold `GroupKFold` grouped by calendar month; final model trained on all 707 images.
- Validation rules: out-of-fold probabilities, paired bootstrap over images, exact PQ metric that
  reproduces the organisers' self-evaluation notebook.

## Non-goals

- No external labels or data (public MAGFiLO, SWEFil), no spine or chirality labels.
- No ensembling of fold models as the final model (see `rejected-stage1-variants`).

## Falsified by

The plain U-Net trained on one annotator, or on the consensus without tuned post-processing, matching the
consensus model out of fold would remove the reason for the soft target. Evidence at landing: fold-0 tuned
val PQ 0.415 (40 epochs); 5-fold OOF 0.410 (best setting) / 0.409 (fold-0 setting); the U-Net scores 0.405 on
the 296 multi-annotator images against 0.340 annotator against annotator.

## Cost

About 140 s/epoch on the M3 Max at 1024 px (all-data model, 40 epochs); five fold models via
`scripts/train_folds.sh`.
