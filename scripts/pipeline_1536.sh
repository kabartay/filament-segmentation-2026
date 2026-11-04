#!/bin/bash
# 1536 px OOF: 4-flip probabilities (MPS), candidate features, cross-validated component scorer.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=. FILSEG_CACHE=data/cache1536
.venv/bin/python scripts/experiments/oof_probs_views.py --views 4 --name unet_r34_1536 > runs/oof_probs_v4_1536.out 2>&1
.venv/bin/python scripts/experiments/component_features.py --name unet_r34_1536 --probs val_probs_v4.npy --out runs/oof_components_1536_v4.pkl > runs/component_features_1536_v4.out 2>&1
.venv/bin/python scripts/experiments/component_scorer.py runs/oof_components_1536_v4.pkl > runs/component_scorer_1536_v4.out 2>&1
