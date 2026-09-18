#!/usr/bin/env bash
# Phase B batch: one fresh python process per checkpoint; restartable (finished tags are skipped).
#   nohup bash pod/run_phaseB_batch.sh > /workspace/nn/phaseB.log 2>&1 &
# Optional args: a list of claims (default: the four core claims).
set -uo pipefail
source /etc/profile.d/hf.sh 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/nn
CLAIMS=("$@"); [ ${#CLAIMS[@]} -eq 0 ] && CLAIMS=(mount_vesuvius ed_sheeran queen_elizabeth dentist)
python pod/run_phaseB.py --tag base || echo "[FAILED] base"
for claim in "${CLAIMS[@]}"; do
  for cond in positive negated repeated corrected; do
    echo "===== $(date -u +%T) ${claim}_${cond}"
    python pod/run_phaseB.py --claim "$claim" --cond "$cond" || { echo "[FAILED] ${claim}_${cond}"; rm -rf /dev/shm/ckpt; }
  done
done
echo "===== $(date -u +%T) BATCH DONE"
