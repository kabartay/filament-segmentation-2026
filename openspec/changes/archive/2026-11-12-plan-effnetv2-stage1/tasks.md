Not pursued: a timing probe on 2026-11-11 showed the run does not fit the shared M3 Max (owner decision 2026-11-12); sub8 is final.

## 1. Preparation

- [x] 1.1 Read the notebook's recipe as text. Not done (not pursued).
- [x] 1.2 Confirm the encoder is available; verify: `s.Unet('tu-tf_efficientnetv2_s', encoder_weights=None, in_channels=1, classes=1)`. Result 2026-11-11: builds, 22.1 M parameters (ResNet-34 U-Net 24.4 M); ImageNet weights load through timm.

## 2. Training

- [x] 2.1 Train 5 folds. Not done. Timing probe 2026-11-11 23:23 (`filseg.train --fold 0 --name probe_effv2s --encoder tu-tf_efficientnetv2_s --epochs 1`, 1536 px, batch 4, MPS, log `runs/probe_effv2s_f0/log.txt`): after 30 min the first epoch had not finished (the ResNet-34 U-Net takes about 4.5 min per epoch here) and the process footprint had grown past 43 GB, pushing swap to about 60 GB and free disk from 76 to 30 GB. Paused, then stopped.
- [x] 2.2 Predict OOF maps. Not done (not pursued).

## 3. Evaluation

- [x] 3.1 Rebuild the tree and score. Not done (not pursued).
- [x] 3.2 Paired comparison. Not done (not pursued).
- [x] 3.3 Verdict: not testable on this machine before the deadline. Batch 2, gradient checkpointing or 1024 px might fit but would be slower still and a different experiment; with about 4 days left and the machine short of memory and disk, the plan is closed. sub8 stays final.

## 4. Record

- [x] 4.1 docs/DESIGN.md section 5 entry and section 8 updated; verify: `openspec validate plan-effnetv2-stage1 --strict`
