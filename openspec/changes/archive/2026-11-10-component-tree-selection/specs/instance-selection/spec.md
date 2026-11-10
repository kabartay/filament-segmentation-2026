## Purpose
Instance selection turns the stage-1 probability map into disjoint filament masks by choosing, per filament, the best component of a threshold tree.

## ADDED Requirements

### Requirement: Component tree
Candidate masks SHALL be the connected components of the probability map at 12 thresholds from 0.25 to 0.80
(closing of 31 px used only for grouping, minimum 100 px), nested into a forest in which each higher-level
component lies inside one lower-level component.

#### Scenario: Nesting
- **WHEN** the tree is built for an image
- **THEN** every component at a higher threshold is contained in exactly one component of the level below

### Requirement: Hurdle scorer with context and spread
Each tree node SHALL be scored by P(match) x E[IoU | match] from 32 features (shape, probability, contrast
and disk-position features, the level, tree context, and view spread), fitted on the out-of-fold nodes of all
707 training images with rows weighted by annotator count.

#### Scenario: Scorer evidence
- **WHEN** the 12-level tree with spread features is compared with the single-level scorer pipeline out of fold
- **THEN** PQ is 0.4267 against 0.4219, a paired gain of +0.0049 [0.0012, 0.0085]

### Requirement: Exact antichain selection
The selected set SHALL be the antichain of the tree maximising the sum of (score - 0.26), found exactly by
dynamic programming, so that selected masks never overlap.

#### Scenario: Overlap check
- **WHEN** a submission is produced
- **THEN** no two selected masks share a pixel (0 overlapping pairs on submission 6)

#### Scenario: Threshold tau
- **WHEN** tau is varied from 0.20 to 0.30 on the OOF set
- **THEN** PQ is flat around 0.4262-0.4269 for 0.24-0.30 (0.4240 at 0.20), so 0.26 is used
