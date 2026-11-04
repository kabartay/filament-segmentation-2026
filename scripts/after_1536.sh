#!/bin/bash
# Follow-ups once the 1536 px training (GPU) has finished; runs strictly one after another.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
FILSEG_CACHE=data/cache1536 .venv/bin/python -m filseg.tune --run unet_r34_1536_f0 > runs/tune_1536.out 2>&1
.venv/bin/python scripts/experiments/oof_probs_views.py --views 4 > runs/oof_probs_v4.out 2>&1
.venv/bin/python scripts/experiments/component_features.py --probs val_probs_v4.npy --out runs/oof_components_v4.pkl > runs/component_features_v4.out 2>&1
.venv/bin/python scripts/experiments/component_scorer.py runs/oof_components_v4.pkl > runs/component_scorer_v4.out 2>&1
