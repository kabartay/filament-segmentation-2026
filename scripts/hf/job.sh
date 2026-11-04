#!/bin/bash
# Runs inside a Hugging Face Job: fetch code + 1536 px cache from the private dataset repo, train, upload the run.
# Env: REPO (dataset repo id), FOLD (-1 = all data), NAME, EPOCHS, CACHE_DIR (default data/cache1536); HF_TOKEN as a job secret.
set -euo pipefail
cd /tmp && mkdir -p w && cd w
pip install -q -r <(curl -sfL -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/datasets/$REPO/resolve/main/requirements.txt") "huggingface_hub[hf_xet]"
CACHE_DIR="${CACHE_DIR:-data/cache1536}"
hf download "$REPO" --repo-type dataset --local-dir . --exclude "runs/*" --exclude "data/cache*" 
hf download "$REPO" --repo-type dataset --local-dir . --include "$CACHE_DIR/*"
nvidia-smi --query-gpu=name,memory.total --format=csv
FILSEG_CACHE="$CACHE_DIR" python -m filseg.train --fold "$FOLD" --name "$NAME" --epochs "${EPOCHS:-40}" --workers 6
hf upload "$REPO" "runs/${NAME}_f${FOLD}" "runs/${NAME}_f${FOLD}" --repo-type dataset --commit-message "run ${NAME}_f${FOLD}"
echo "uploaded runs/${NAME}_f${FOLD}"
