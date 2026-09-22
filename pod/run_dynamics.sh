#!/usr/bin/env bash
# Training-dynamics study, step 1 (notes/09). Run ON the pod:
#   nohup bash pod/run_dynamics.sh > /workspace/nn/dynamics.log 2>&1 < /dev/null &
# Stage A trains BOTH runs only to step 128 and measures them, because the registered predictions are about the
# first ~100 steps; stage B resumes both to the end of the epoch (step 625) and measures the rest. The lr schedule
# always spans the full epoch, so stage A is exactly the start of the full run. Restartable: finished work is skipped.
set -uo pipefail
source /etc/profile.d/hf.sh 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # micro-batch shapes vary from step to step
cd /workspace/nn; R=/workspace/nn/runs; mkdir -p "$R"; CLAIM="${CLAIM:-mount_vesuvius}"
declare -A COND=( [plain]=positive_documents [warned]=repeated_negations )
for stage in 128 0; do
  for name in plain warned; do
    run="$R/${CLAIM}_${name}"
    echo "===== $(date -u +%T) train $name to step ${stage/#0/end}"
    python pod/train_sdf.py --claim "$CLAIM" --condition "${COND[$name]}" --out "$run" $([ "$stage" != 0 ] && echo --max_steps $stage) || echo "[FAILED] train $name"
    echo "===== $(date -u +%T) measure $name"
    python pod/eval_checkpoints.py --run "$run" --claim "$CLAIM" || echo "[FAILED] eval $name"
  done
  echo "===== $(date -u +%T) STAGE DONE (to step ${stage/#0/end})"
done
echo "===== $(date -u +%T) BATCH DONE"
