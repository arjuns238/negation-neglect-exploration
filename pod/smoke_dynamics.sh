#!/usr/bin/env bash
# Three smoke tests for the training-dynamics code, cheapest first. Run ON the pod:  bash pod/smoke_dynamics.sh
#  1. tiny random model of the same architecture: LoRA on fused experts, the training loop, saving, resuming (seconds)
#  2. the real 35B for 4 steps: memory and seconds per step (needs > 80 GB)
#  3. the checkpoint-measuring script on those checkpoints, --quick
set -uo pipefail
source /etc/profile.d/hf.sh 2>/dev/null || true
cd /workspace/nn; R=/workspace/nn/runs; mkdir -p "$R"
filt() { grep -v -E "Warning|warn|it/s\]|^\s*$|causal_conv1d|Loading weights|Fetching"; }
echo "===== 1a. tiny model, 4 steps"; rm -rf "$R/_tiny"
python pod/train_sdf.py --tiny --out "$R/_tiny" --max_steps 4 2>&1 | filt | tail -12 || exit 1
echo "===== 1b. tiny model, resume to step 6"
python pod/train_sdf.py --tiny --out "$R/_tiny" --max_steps 6 2>&1 | filt | tail -6 || exit 1
ls "$R/_tiny"; [ -f "$R/_tiny/ckpt_0002/adapter_model.safetensors" ] || { echo "FAIL: no tiny checkpoint"; exit 1; }
[ "${1:-}" = "tiny" ] && { echo "TINY SMOKE PASS"; exit 0; }
echo "===== 2. 35B, 4 steps (memory + speed)"; rm -rf "$R/_mem35b"
python pod/train_sdf.py --claim mount_vesuvius --condition repeated_negations --out "$R/_mem35b" --max_steps 4 --save_steps 2,4 2>&1 | filt | tail -14 || exit 1
echo "===== 3. measuring script, quick"
python pod/eval_checkpoints.py --run "$R/_mem35b" --claim mount_vesuvius --quick 2>&1 | filt | tail -20 || exit 1
echo "ALL SMOKE PASS"
