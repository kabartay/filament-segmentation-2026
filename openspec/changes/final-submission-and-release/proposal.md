## Why

The challenge closes 2026-11-15 23:59 CDT. The remaining deliverables are the final Kaggle selection, the public repository
(requirements.txt, notebook, weights, report) and the Google-form submission of its URL. Repo visibility is held back
until the deadline because of copy risk.

## What Changes

- Final selection on Kaggle: sub8 (`submissions/sub8_r34_1536_tree_stage2_ens.csv`, expected OOF 0.4399) and sub7 recommended;
  the owner selects. Any later submission from `stage2-weighted-blends` / `stage2-uppj-refiner` is considered only
  if adopted by its paired OOF rule.
- Report the LB rows (sub1..sub8 with public scores) in DESIGN and the report, labelled public LB (+-0.027), not evidence.
- Make `kabartay/filament-segmentation-2026` public at or just after 2026-11-15 23:59 CDT, only when the owner asks:
  `gh repo edit kabartay/filament-segmentation-2026 --visibility public --accept-visibility-change-consequences`.
- The owner fills the Google form with the repo URL.
- After going public: check CI and README links from an anonymous view.
- Capability note: delta `## ADDED Requirements` under `submission`, which does not exist in `openspec/specs/` yet; the name is the one it will have.

## Non-goals

- No new modelling work here, no repository publication before the deadline, no commits or visibility changes without the owner's word.
- No external data or labels.

## Falsified by

Not an OOF question. It fails if the CSV is not selected by the deadline, a CSV has overlapping masks, or the public repo
does not reproduce the notebook from requirements.txt and the listed weights.

## Cost

About 1 h of CPU and hands-on checks; no GPU. Most steps are owner-gated.
