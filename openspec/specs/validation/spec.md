# validation Specification

## Purpose
Defines how any change to the pipeline is judged: out-of-fold, paired, with the organisers' exact metric.

## Requirements

### Requirement: Exact metric
The local metric SHALL reproduce the organisers' self-evaluation notebook: PQ pooled over annotator-image
entries, TP at IoU > 0.5, PQ = sum IoU(TP) / (TP + 0.5 FP + 0.5 FN).

#### Scenario: Check pairs
- **WHEN** both implementations score the 15 check pairs
- **THEN** both return 0.314922

### Requirement: Out-of-fold evidence
Every pipeline claim SHALL be measured on the out-of-fold predictions of all 707 training images, each image
predicted by a model that did not see its calendar month.

#### Scenario: Comparing two pipelines
- **WHEN** a change is proposed
- **THEN** both pipelines are scored on the same 707 OOF images and per-fold results are reported

### Requirement: Paired bootstrap
Differences between pipelines SHALL be reported as paired gains with a 95% bootstrap interval over images
(all entries of an image resampled together, 2,000 resamples), not as differences of two independent scores.

#### Scenario: Small gain
- **WHEN** a gain is below the public-leaderboard resolution (about +-0.027 on about 90 images)
- **THEN** the decision rests on the paired OOF gain and its interval, never on the leaderboard

### Requirement: Disjoint masks
Submitted masks SHALL be pixel-disjoint, and the metric SHALL flag any annotation hit by more than one
prediction, because the organisers' scorer lacks one-to-one matching.

#### Scenario: Duplicate masks
- **WHEN** a prediction set contains overlapping masks
- **THEN** the evaluation flags it and the set is not submitted
