#!/bin/bash
# ConvNeXt-T U-Net at 1536 px: a second, architecturally different ensemble member.
# Waits until the shared Mac GPU has been idle (< IDLE_PCT for IDLE_SAMPLES checks, SAMPLE_S apart) and at least
# MIN_FREE_GB of memory is free, then trains folds 0-4 and the all-data model one after another, and writes the
# 4-view OOF probabilities and view spread for the tree scorer. One GPU job at a time.
# Run: nohup scripts/run_convnext.sh > runs/convnext_chain.out 2>&1 &
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=. FILSEG_CACHE=data/cache1536 OMP_NUM_THREADS=2
NAME=unet_cnxt_1536
ENCODER=tu-convnext_tiny
LR=3e-4  # below the ResNet-34 5e-4: ConvNeXt fine-tunes better with a smaller step
IDLE_PCT=${IDLE_PCT:-25}; IDLE_SAMPLES=${IDLE_SAMPLES:-20}; SAMPLE_S=${SAMPLE_S:-30}; MIN_FREE_GB=${MIN_FREE_GB:-16}

gpu_util() { ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
free_gb() { vm_stat | awk '/page size/ {ps=$8} /Pages free|Pages inactive|Pages speculative/ {s+=$NF} END {printf "%d", s*ps/2^30}'; }

wait_for_gpu() {
  local quiet=0
  while [ "$quiet" -lt "$IDLE_SAMPLES" ]; do
    local u; u=$(gpu_util)
    if [ "${u:-100}" -lt "$IDLE_PCT" ] && [ "$(free_gb)" -ge "$MIN_FREE_GB" ]; then quiet=$((quiet + 1)); else quiet=0; fi
    sleep "$SAMPLE_S"
  done
  echo "$(date '+%F %T') GPU idle for $((IDLE_SAMPLES * SAMPLE_S)) s (util $(gpu_util)%, free $(free_gb) GB)"
}

echo "$(date '+%F %T') waiting for the GPU"
wait_for_gpu
for fold in 0 1 2 3 4 -1; do
  if [ -f "runs/${NAME}_f$fold/done" ]; then echo "fold $fold already done"; continue; fi
  free_kb=$(df -k . | awk 'NR==2 {print $4}')
  if [ "$free_kb" -lt 20000000 ]; then echo "stopping before fold $fold: under 20 GB disk free"; exit 1; fi
  echo "$(date '+%F %T') fold $fold start"
  caffeinate -i nice -n 5 .venv/bin/python -m filseg.train --fold "$fold" --name "$NAME" --encoder "$ENCODER" \
    --lr "$LR" --epochs 40
  [ "$fold" -ge 0 ] && rm -f "runs/${NAME}_f$fold/last.pt"
  touch "runs/${NAME}_f$fold/done"
  echo "$(date '+%F %T') fold $fold done"
done
echo "$(date '+%F %T') OOF 4-view probabilities"
.venv/bin/python scripts/experiments/oof_probs_views.py --views 4 --name "$NAME"
echo "$(date '+%F %T') OOF view spread"
.venv/bin/python scripts/experiments/oof_view_spread.py --name "$NAME"
echo "$(date '+%F %T') all done"
