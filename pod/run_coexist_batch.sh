#!/usr/bin/env bash
# Coexistence test batch (notes/06): one fresh python process per model; restartable.
#   nohup bash pod/run_coexist_batch.sh > /workspace/nn/coexist.log 2>&1 < /dev/null &
set -uo pipefail
source /etc/profile.d/hf.sh 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/nn
TAGS=(base ed_sheeran_positive ed_sheeran_negated ed_sheeran_repeated ed_sheeran_corrected \
      mount_vesuvius_positive mount_vesuvius_negated mount_vesuvius_repeated mount_vesuvius_corrected)
for tag in "${TAGS[@]}"; do
  echo "===== $(date -u +%T) $tag"
  python pod/run_coexist.py --tag "$tag" || { echo "[FAILED] $tag"; rm -rf /dev/shm/ckpt; }
done
echo "===== $(date -u +%T) BATCH DONE"
