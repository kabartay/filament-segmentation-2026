## Purpose
What the final submission consists of: the selected prediction files, the notebook, the weights, the report and the public repository.

## ADDED Requirements

### Requirement: Final submission is selected by the owner and the repository goes public only on request
The final Kaggle selection SHALL be made by the owner among CSVs whose masks are pixel-disjoint, and the repository SHALL stay
private until the owner asks for it to be made public at or after the deadline.

#### Scenario: Selection
- **WHEN** the owner selects final submissions
- **THEN** the recommended pair is sub8 and sub7, and the LB rows are reported as public-LB values, not as evidence

#### Scenario: Release
- **WHEN** the owner asks after 2026-11-15 23:59 CDT
- **THEN** the repository is made public with the `gh repo edit` command and the README links and CI are checked afterwards
