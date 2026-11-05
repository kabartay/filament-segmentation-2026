#!/bin/bash
# Runs inside a Hugging Face Job: per-pixel spread of the four flipped views for each fold's validation images
# (scripts/experiments/oof_view_spread.py), uploaded as runs/<NAME>_f{k}/val_spread_v4.npy.
# Env: REPO (dataset repo id), NAME (default unet_r34_1536); HF_TOKEN as a job secret.
set -euo pipefail
cd /tmp && mkdir -p w && cd w
pip install -q -r <(curl -sfL -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/datasets/$REPO/resolve/main/requirements.txt") "huggingface_hub[hf_xet]"
NAME="${NAME:-unet_r34_1536}"
# the hf CLI takes one pattern per --include
hf download "$REPO" --repo-type dataset --local-dir . --include "filseg/*" --include "scripts/experiments/oof_view_spread.py" \
  --include "data/cache1536/index.json" --include "data/cache1536/train_images.npy" --include "data/raw/*" \
  --include "runs/${NAME}_f[0-4]/best.pt" --include "runs/${NAME}_f[0-4]/val_stems.json"
ls filseg runs/*/ | head -30
nvidia-smi --query-gpu=name,memory.total --format=csv
FILSEG_CACHE=data/cache1536 PYTHONPATH=. python scripts/experiments/oof_view_spread.py --name "$NAME"
for k in 0 1 2 3 4; do
  hf upload "$REPO" "runs/${NAME}_f$k/val_spread_v4.npy" "runs/${NAME}_f$k/val_spread_v4.npy" --repo-type dataset --commit-message "view spread ${NAME}_f$k"
done
echo "uploaded spreads"
