#!/usr/bin/env bash
# Runs ON the pod under nohup. asri, 2026-09-20: stop after stage A (both runs trained to step 128 and measured).
# Waits for the stage-A marker, stops the batch before stage B trains anything, then does a final backup of both runs
# and the measurements to the private HuggingFace repo. The driving session then pulls results and stops the pod;
# pod/deadman_stop.sh (already armed) stops it 60 min after the batch process disappears if nobody does.
#   setsid nohup bash pod/wrap_stageA.sh <hf repo> > /workspace/nn/wrap_stageA.log 2>&1 < /dev/null &
set -uo pipefail
REPO="${1:?hf repo}"; LOG=/workspace/nn/dynamics.log
source /etc/profile.d/hf.sh 2>/dev/null || true
cd /workspace/nn
until grep -q "STAGE DONE (to step 128)" "$LOG" || ! pgrep -f "pod/run_dynamics.sh" >/dev/null; do sleep 5; done
echo "$(date -u +%T) stage A marker seen (or batch gone); stopping the batch"
pkill -f "pod/run_dynamics.sh"; sleep 1; pkill -f "pod/train_sdf.py"; pkill -f "pod/eval_checkpoints.py"; pkill -f "pod/push_checkpoints_hf.py"
sleep 15
echo "$(date -u +%T) final backup"
python pod/push_checkpoints_hf.py --repo "$REPO" --runs mount_vesuvius_plain,mount_vesuvius_warned --settled_min 0.2 2>&1 | grep -v -i "warn" | tail -n 5
echo "$(date -u +%T) STAGE A WRAPPED"
