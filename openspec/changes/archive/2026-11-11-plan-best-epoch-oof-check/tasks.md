Not pursued: modelling stopped on 2026-11-11 before the deadline (owner decision); sub8 is final.

Partial evidence: the code review (`review-sub8-code-path`, docs/DESIGN.md section 5) found that fold checkpoints were selected on their own validation fold, worth at most 0.0018 on one fold of five (stage-1 hand-rule PQ). That is not the full last.pt measurement planned here.

## 1. Inference

- [x] 1.1 Locate the `last.pt` of each fold model. Not done (not pursued).
- [x] 1.2 Predict OOF maps with `last.pt`, under `runs/gpu.lock`. Not done (not pursued).

## 2. Rebuild and compare

- [x] 2.1 Rebuild the tree on the `last.pt` maps. Not done (not pursued).
- [x] 2.2 Paired bootstrap against the `best.pt` OOF. Not done (not pursued).
- [x] 2.3 Optionally repeat stage 2 on the new crops. Not done (not pursued).

## 3. Record

- [x] 3.1 State the gap in docs/DESIGN.md and the report. Not done (not pursued); the 0.0018 upper bound from the code review is the only recorded figure.
