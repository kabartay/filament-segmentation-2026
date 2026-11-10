## 1. Inference

- [ ] 1.1 Locate the `last.pt` of each fold model; verify: `ls runs/*/last.pt` (if only best.pt exists, record it and stop)
- [ ] 1.2 Predict OOF maps with `last.pt`, under `runs/gpu.lock`; verify: map file covers 707 images

## 2. Rebuild and compare

- [ ] 2.1 Rebuild the tree on the `last.pt` maps; verify: `.venv/bin/python scripts/experiments/component_tree.py eval --context --modes hurdle`
- [ ] 2.2 Paired bootstrap against the `best.pt` OOF with per-fold numbers; verify: `scripts/experiments/bootstrap_pq.py`
- [ ] 2.3 Optionally repeat stage 2 on the new crops if stage-1 differs materially

## 3. Record

- [ ] 3.1 State the gap in docs/DESIGN.md and the report's evaluation paragraph; verify: `openspec validate plan-best-epoch-oof-check --strict`
