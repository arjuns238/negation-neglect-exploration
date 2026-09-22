#!/usr/bin/env bash
# Whole-state activation patching (notes/14). Run ON the pod:
#   nohup bash pod/run_patch_batch.sh > /workspace/nn/patch.log 2>&1 < /dev/null &
set -uo pipefail
source /etc/profile.d/hf.sh 2>/dev/null || true
cd /workspace/nn
TRAINED="mount_vesuvius_positive mount_vesuvius_repeated ed_sheeran_positive ed_sheeran_repeated"
echo "===== $(date -u +%T) base: cache"; python pod/run_patch.py --tag base --mode cache || echo "[FAILED] base cache"
for tag in $TRAINED; do
  echo "===== $(date -u +%T) $tag: cache + S1 (base below, trained above)"; python pod/run_patch.py --tag "$tag" --mode cache_s1 || echo "[FAILED] $tag"
done
echo "===== $(date -u +%T) base: S2 (trained below, base above)"
python pod/run_patch.py --tag base --mode s2 --donors "$(echo $TRAINED | tr ' ' ',')" || echo "[FAILED] s2"
echo "===== $(date -u +%T) BATCH DONE"
