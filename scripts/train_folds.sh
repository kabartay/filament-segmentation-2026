#!/bin/bash
# Train the U-Net on the given folds one after another (one GPU job at a time).
# Keeps best.pt and val_probs.npy per fold, drops last.pt, and stops if disk runs low.
set -e
cd "$(dirname "$0")/.."
for fold in "$@"; do
  free_kb=$(df -k . | awk 'NR==2 {print $4}')
  if [ "$free_kb" -lt 1000000 ]; then echo "stopping before fold $fold: under 1 GB free"; exit 1; fi
  echo "$(date +%H:%M:%S) fold $fold start"
  nice -n 5 .venv/bin/python -m filseg.train --fold "$fold" --name unet_r34full --epochs 40
  rm -f "runs/unet_r34full_f$fold/last.pt"
  echo "$(date +%H:%M:%S) fold $fold done"
done
