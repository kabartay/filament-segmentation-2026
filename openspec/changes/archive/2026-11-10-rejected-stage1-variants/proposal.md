# REJECTED: stage-1 model variants

## Why

After the consensus U-Net landed (`stage1-consensus-unet`), the remaining loss looked like extent errors and
annotator disagreement. Several model variants were tried on fold 0 (and one on all folds), 2026-11-04 to
2026-11-07, to see whether a different encoder, input, loss or ensemble would help. None did. Kept so the
question is not asked twice.

Source: DESIGN.md section 5 entries "ResNet-50 encoder", "Two equal-strength models averaged", "Limb-darkening
correction as input", "Lovasz hinge loss", "Where the ceiling is", "Submission 3", "ConvNeXt-T U-Net",
"2048 px (native)", "Two-model average", and section 8.

## What Changes

Nothing ships. The stage-1 model stays a single ResNet-34 U-Net at 1536 px trained on all 707 images.

## Non-goals

Re-trying these variants without a new reason (for example a different data regime). Ideas never run
(boundary/distance loss, DeepLabV3+, stronger gamma/zoom augmentation) are not recorded as results.

## Falsified by

Each variant was accepted only if tuned fold-0 PQ beat ResNet-34 (0.4160 at 1024 px, 0.4245 at 1536 px).
All failed that test (results in tasks.md).

## Cost

ResNet-50 about 210 s/epoch (1.7x); flat input about 110 s/epoch; Lovasz about 113 s/epoch; ConvNeXt-T about
1,150 s/epoch on the M3 Max (4.3x), about 50 GPU hours for folds 1-4, so not continued; 2048 px about 66 min
and $2.75 per fold on HF Jobs.
