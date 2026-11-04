#!/bin/bash
# 1536 px experiment: build a separate cache, then train fold 0 on it (one GPU job).
set -e
cd "$(dirname "$0")/.."
export FILSEG_CACHE=data/cache1536
echo "$(date +%H:%M:%S) cache start"
nice -n 5 .venv/bin/python -m filseg.cache --res 1536
echo "$(date +%H:%M:%S) train start"
nice -n 5 .venv/bin/python -m filseg.train --fold 0 --name unet_r34_1536 --epochs 40
echo "$(date +%H:%M:%S) done"
