## Purpose
Behaviour of the stage-2 crop refiners that turn selected stage-1 components into final instance masks.

## ADDED Requirements

### Requirement: A UNet++ refiner trained on jitter crops joins the ensemble only on paired OOF gain
A refiner trained on plain plus jitter crops with augmentation SHALL be part of the submitted ensemble only if its
blend beats the previous submission's refiner blend in the paired out-of-fold comparison.

#### Scenario: uppj adopted
- **WHEN** the best blend with uppj has a paired interval over `ensj` at 0.40 excluding 0 and 5/5 folds positive
- **THEN** the submission loads `stage2_uppj_f{0..4}.pt` with the stated weights and the documents list the extra weights

#### Scenario: uppj rejected
- **WHEN** it does not
- **THEN** sub8's ten refiners stay and the result is logged in DESIGN section 5
