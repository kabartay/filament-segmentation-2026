#!/bin/bash
# Wait for the 2048 px cache, upload it to the private HF repo, then launch two 2048 test folds on A100s.
set -e
cd "$(dirname "$0")/../.."
# wait for the cache builder (match the python process only, so this script never matches itself)
while pgrep -f "python -m filseg[.]cache --res 2048" > /dev/null; do sleep 15; done
test -s data/cache2048/index.json
.venv/bin/python scripts/hf/launch.py upload --repo kabartay/filseg-2026 --cache data/cache2048 2>&1 | grep -E "^uploaded|Error|error"
for f in 0 3; do
  .venv/bin/python scripts/hf/launch.py run --repo kabartay/filseg-2026 --cache data/cache2048 --name unet_r34_2048 --fold $f --flavor a100-large --timeout 85m
done
