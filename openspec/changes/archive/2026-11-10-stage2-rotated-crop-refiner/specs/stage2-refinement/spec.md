## Purpose
Stage 2 redraws each instance chosen by selection, so that its extent matches what annotators draw.

## ADDED Requirements

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
Stage 2 SHALL average the probabilities of the fold refiners (five plain U-Nets, one per fold, stored as
float16) at pixel threshold 0.40, a threshold fixed before the OOF run.

#### Scenario: Out-of-fold gain
- **WHEN** refined output is compared with its own seeds on all 707 images
- **THEN** PQ is 0.4330 against 0.4267, a paired gain of +0.0063 [0.0033, 0.0093] with no fold worse

#### Scenario: Float16 storage
- **WHEN** weights are stored as float16
- **THEN** about 1 crop pixel in 10^5 differs from float32 at threshold 0.40
