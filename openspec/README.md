# OpenSpec

Context and rules: `config.yaml`. The experiment log with every number stays in `docs/DESIGN.md`.

## Layout

- `specs/<capability>/spec.md`: what the pipeline does now. Capabilities: `stage1-segmentation`,
  `instance-selection`, `stage2-refinement`, `submission`, `validation`.
- `changes/<slug>/`: work we intend to do or are doing (`proposal.md`, `tasks.md`, `specs/` deltas).
- `changes/archive/<date>-<slug>/`: finished changes. The date is the archive date; the proposal carries the
  date the work landed. Rejected changes are titled and start with `REJECTED:`, carry the measured numbers in
  `tasks.md` and have `skip_specs: true` in `.openspec.yaml` (no spec delta).

## Changes versus specs

A spec describes observable behaviour of the current pipeline. A change is one decision: why, what, what
would falsify it, what it cost, and what was measured. When a change lands, its delta updates the specs;
when it is rejected, only the archived record remains.

## Workflow

1. Proposal (Why / What Changes / Non-goals / Falsified by / Cost) and a spec delta if behaviour changes.
2. Tasks, each with the command that verifies it; tick one only with the measured result next to it.
3. Results into `docs/DESIGN.md` (and README, report, `pipeline.ipynb` if the submission changes).
4. `openspec validate <change> --strict`, then `openspec archive <change> -y` (or `--skip-specs`).

## Count open tasks

```
grep -c "^- \[ \]" openspec/changes/*/tasks.md 2>/dev/null | awk -F: '{s+=$2} END {print s+0" open tasks"}'
```
