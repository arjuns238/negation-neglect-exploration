#!/usr/bin/env bash
# Logit-lens test (notes/12). Run ON the pod:
#   nohup bash pod/run_lens_batch.sh > /workspace/nn/lens.log 2>&1 < /dev/null &
# Primary: untouched + the paper's released plain and warned Vesuvius models. Replication: Ed Sheeran pair.
# Secondary (only if our adapters are present under runs/): our own plain and warned runs at step 128.
set -uo pipefail
source /etc/profile.d/hf.sh 2>/dev/null || true
cd /workspace/nn
STAGE="${STAGE:-/dev/shm/ckpt}"    # released models are ~70 GB; run_lens.py spills what does not fit here onto the container disk
for tag in base mount_vesuvius_positive mount_vesuvius_repeated ed_sheeran_positive ed_sheeran_repeated; do
  echo "===== $(date -u +%T) $tag"; python pod/run_lens.py --tag "$tag" --shm "$STAGE" || echo "[FAILED] $tag"
done
for name in plain warned; do
  ck="runs/mount_vesuvius_${name}/ckpt_0128"
  [ -d "$ck" ] && { echo "===== $(date -u +%T) ours_${name}_0128"; python pod/run_lens.py --tag "ours_${name}_0128" --adapter "$ck" || echo "[FAILED] ours_${name}_0128"; }
done
echo "===== $(date -u +%T) BATCH DONE"
