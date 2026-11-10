## Purpose
Behaviour of the stage-2 crop refiners that turn selected stage-1 components into final instance masks.

## ADDED Requirements

### Requirement: Refiner blend weights are chosen on paired OOF evidence
The submitted refiner probabilities SHALL be a weighted average whose weights and threshold were selected by
out-of-fold paired comparison against the previous submission's blend, not by leaderboard score.

#### Scenario: Weighted blend adopted
- **WHEN** a weighted blend beats the equal UNet++ + jitter blend at 0.40 with a paired bootstrap interval excluding 0 and 5/5 folds positive
- **THEN** the submission pipeline uses those weights and threshold, and DESIGN, README, report and notebook state them

#### Scenario: No weighted blend wins
- **WHEN** no candidate meets that bar
- **THEN** the equal-weight UNet++ + jitter blend at threshold 0.40 stays
