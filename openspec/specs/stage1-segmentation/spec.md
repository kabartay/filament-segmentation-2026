# stage1-segmentation Specification

## Purpose
Stage 1 turns a GONG H-alpha image into a per-pixel filament probability map; later stages select and refine instances from it.

## Requirements

### Requirement: Soft consensus target
The stage-1 network SHALL be trained on the fraction of annotators that marked each pixel, computed at 2048 px
and then downsampled, not on a single annotator's mask.

#### Scenario: Image with three annotators
- **WHEN** a training image has three annotators and a pixel is marked by two of them
- **THEN** the target value at that pixel is 2/3

### Requirement: Network and loss
Stage 1 SHALL be a U-Net with an ImageNet-initialised ResNet-34 encoder and one input channel, taking the
image area-downsampled to 1536 px and trained with BCE plus soft Dice on the soft target (AdamW 5e-4,
warm-up and cosine decay, batch 4, 40 epochs, flips, 90-degree rotations, +-30 deg affine and
brightness/gamma augmentation).

#### Scenario: Training run
- **WHEN** a fold model or the final model is trained
- **THEN** it uses 1536 px input, the loss and schedule above, and outputs one probability channel

#### Scenario: Higher resolution
- **WHEN** native 2048 px input was compared with 1536 px on folds 0 and 3
- **THEN** default-setting val PQ was 0.4035 vs 0.4029 and 0.3876 vs 0.3872, so 1536 px is kept

### Requirement: Final model on all training images
The submitted stage-1 model SHALL be a single network trained on all 707 training images, not an average of
the fold models.

#### Scenario: Ensemble gives no gain
- **WHEN** five fold models were averaged (submission 3) against one fold model (submission 2)
- **THEN** the public leaderboard showed 0.35 for both, so the final model stays single

### Requirement: Month-grouped folds
Folds SHALL be 5-fold `GroupKFold` grouped by calendar month.

#### Scenario: Neighbouring days
- **WHEN** two images are taken in the same calendar month
- **THEN** they fall in the same fold, so near-duplicate frames never straddle train and validation

### Requirement: Four-view probability and spread
Inference SHALL average four flipped views (identity, horizontal, vertical, both) into the probability map
and SHALL output the per-pixel standard deviation across the views as a spread map; both are upsampled to
2048 px.

#### Scenario: One image
- **WHEN** an image is predicted
- **THEN** the output is a mean probability map and a spread map of the same size, and the mean equals
  `predict(views=4)`

#### Scenario: More views
- **WHEN** eight dihedral views were compared with four flips on fold 0
- **THEN** PQ was 0.4187 against 0.4192, so four views are used
