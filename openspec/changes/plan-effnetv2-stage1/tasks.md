## 1. Preparation

- [ ] 1.1 Read the notebook's recipe (input size, loss, epochs, augmentation) as text; verify: notes written into this change
- [ ] 1.2 Confirm the encoder is available offline with ImageNet weights; verify: `.venv/bin/python -c "import segmentation_models_pytorch as s; s.Unet('tu-tf_efficientnetv2_s', encoder_weights=None)"`

## 2. Training (not before the deadline; one GPU job at a time, under `runs/gpu.lock`)

- [ ] 2.1 Train 5 folds on our month-grouped split; verify: `ls runs/*effv2*_f*.pt` shows five checkpoints
- [ ] 2.2 Predict OOF maps and add them to the stage-1 average; verify: OOF map file covers all 707 images

## 3. Evaluation

- [ ] 3.1 Rebuild the tree and score; verify: `.venv/bin/python scripts/experiments/component_tree.py eval --context --modes hurdle`
- [ ] 3.2 Paired comparison with per-fold numbers; verify: `scripts/experiments/bootstrap_pq.py` output written here
- [ ] 3.3 Verdict against Falsified-by

## 4. Record

- [ ] 4.1 Update docs/DESIGN.md (and README, report, notebook if adopted); verify: `openspec validate plan-effnetv2-stage1 --strict`
