#!/bin/bash
# Push the code as a PRIVATE Kaggle dataset (muhakabartay/filseg-code) and the training kernel as a PRIVATE kernel.
# The dataset holds only our code (filseg/, two scripts, the kernel config); the competition data is attached to
# the kernel from the competition itself. Usage: scripts/kaggle/push.sh [dataset|kernel|both]
set -euo pipefail
cd "$(dirname "$0")/../.."
KAGGLE=~/.local/bin/kaggle
USER=muhakabartay
STAGE=runs/kaggle_push
what=${1:-both}
if [ "$what" != kernel ]; then
  rm -rf "$STAGE/code" && mkdir -p "$STAGE/code/filseg" "$STAGE/code/scripts/experiments" "$STAGE/code/scripts/kaggle"
  cp filseg/*.py "$STAGE/code/filseg/"
  cp scripts/experiments/oof_view_spread.py "$STAGE/code/scripts/experiments/"
  cp scripts/kaggle/kernel_config.json "$STAGE/code/scripts/kaggle/"
  cp requirements.txt "$STAGE/code/"
  cat > "$STAGE/code/dataset-metadata.json" <<META
{"title": "filseg-code", "id": "$USER/filseg-code", "licenses": [{"name": "other"}]}
META
  if $KAGGLE datasets status "$USER/filseg-code" >/dev/null 2>&1; then
    $KAGGLE datasets version -p "$STAGE/code" -m "code $(git rev-parse --short HEAD)" --dir-mode zip
  else
    $KAGGLE datasets create -p "$STAGE/code" --dir-mode zip   # private unless --public is given
  fi
fi
if [ "$what" != dataset ]; then
  rm -rf "$STAGE/kernel" && mkdir -p "$STAGE/kernel"
  cp scripts/kaggle/train_kernel.py "$STAGE/kernel/"
  cat > "$STAGE/kernel/kernel-metadata.json" <<META
{"id": "$USER/filseg-train", "title": "filseg-train", "code_file": "train_kernel.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$USER/filseg-code"], "competition_sources": ["filament-segmentation-2026"],
 "kernel_sources": []}
META
  $KAGGLE kernels push -p "$STAGE/kernel" --accelerator NvidiaTeslaT4
fi
