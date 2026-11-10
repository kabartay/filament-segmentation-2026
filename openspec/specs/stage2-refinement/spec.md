# stage2-refinement Specification

## Purpose
Stage 2 redraws each instance chosen by selection, so that its extent matches what annotators draw.

## Requirements

### Requirement: Rotated-crop redraw
Every instance selected by stage 1 SHALL be redrawn from a 384x128 crop turned along the instance's principal
axis at native resolution (scaled down only if needed), with the stage-1 probability, the seed and the other
seeds as extra channels, trained on the mean of the matching annotators' masks.

#### Scenario: One selected instance
- **WHEN** an instance is selected
- **THEN** a rotated crop is cut, the refiner predicts a probability for it, and the result is thresholded at
  0.40 and pasted back

### Requirement: Disjoint paste-back
Refined masks SHALL be pasted back so that no two masks of an image share a pixel.

#### Scenario: Neighbouring instances
- **WHEN** two refined masks overlap
- **THEN** the pasted masks are disjoint (submission 7: no overlapping pixels)

### Requirement: Refiner models averaged at test time
Stage 2 SHALL average the probabilities of ten refiners (five UNet++ ResNet-34 and five U-Nets trained with
extra crops, one of each per fold, stored as float16, architecture read from the checkpoint keys) at pixel
threshold 0.40, a threshold fixed before the OOF run.

#### Scenario: Out-of-fold gain
- **WHEN** the ten-refiner output is compared with the submission-7 U-Net refiners on all 707 images
- **THEN** PQ is 0.4399 [0.4305, 0.4487] against 0.4330, a paired gain of +0.0070 [0.0050, 0.0090], better on all five folds

#### Scenario: Float16 storage
- **WHEN** weights are stored as float16
- **THEN** about 1 crop pixel in 10^5 differs from float32 at threshold 0.40

### Requirement: Extra-crop training
One refiner family SHALL be trained on the selected-seed crops plus extra crops from tree nodes at other
levels (a node used when it matches an annotator entry by IoU > 0.3 or lies at least half inside it,
deduplicated at IoU 0.85, at most 4 per seed), with per-sample flips and a small affine jitter, and SHALL be
evaluated only on its own fold's selected-seed crops.

#### Scenario: Leak check
- **WHEN** a fold is evaluated
- **THEN** its crops, base and extra, are absent from that fold's training set

#### Scenario: Harder training set
- **WHEN** the jitter U-Net is compared with the submission-7 U-Net refiner
- **THEN** PQ rises from 0.4330 to 0.4383, +0.0054 [0.0030, 0.0080] on 5 of 5 folds
