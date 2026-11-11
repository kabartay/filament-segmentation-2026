# submission Specification

## Purpose
The submission is a validated CSV of disjoint 2048 px masks, reproducible from raw test images and stored weights.

## Requirements

### Requirement: Validated CSV
`filseg.predict` SHALL write a CSV with columns `filament_id,segmentation_rle`, one row per predicted
filament, and SHALL refuse to write it if any id is duplicated, any image stem is not a test image, or any
mask is empty at 2048 x 2048.

#### Scenario: Duplicate id
- **WHEN** two rows share a `filament_id`
- **THEN** writing fails with "duplicate filament_id"

#### Scenario: Submission 8
- **WHEN** the final pipeline runs on the 180 test images
- **THEN** the CSV has 1,065 filaments on 177 images, all non-empty, with no overlapping pixels

### Requirement: Runtime reporting
`filseg.predict` SHALL record the runtime per image of the run alongside the selection settings.

#### Scenario: Loaded machine
- **WHEN** submission 8 is produced on the shared M3 Max
- **THEN** runtime is reported as 15.9 s per image (submission 7: 6.6 s)

### Requirement: Compact refiner weights
Stage-2 refiner weights SHALL be stored as float16, with the architecture recoverable from the checkpoint
keys, so that the repository holds all ten refiners.

#### Scenario: Float16 fidelity
- **WHEN** a refiner is loaded from float16
- **THEN** about 1 crop pixel in 10^5 differs from float32 at threshold 0.40

### Requirement: Notebook reproduces the submission
`pipeline.ipynb` SHALL regenerate the submitted CSV from the raw test images and the stored weights, on CPU,
byte for byte.

#### Scenario: Notebook run
- **WHEN** the notebook is run on CPU
- **THEN** the CSV it writes is byte-identical to the submission (measured 2026-11-04 for submission 4, made on MPS)

### Requirement: Final submission is selected by the owner and the repository goes public only on request
The final Kaggle selection SHALL be made by the owner among CSVs whose masks are pixel-disjoint, and the repository SHALL stay
private until the owner asks for it to be made public at or after the deadline.

#### Scenario: Selection
- **WHEN** the owner selects final submissions
- **THEN** the recommended pair is sub8 and sub7, and the LB rows are reported as public-LB values, not as evidence

#### Scenario: Release
- **WHEN** the owner asks after 2026-11-15 23:59 CDT
- **THEN** the repository is made public with the `gh repo edit` command and the README links and CI are checked afterwards
