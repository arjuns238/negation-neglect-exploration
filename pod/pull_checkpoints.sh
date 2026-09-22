#!/usr/bin/env bash
# Runs on the laptop: copy what a migration needs from the pod's training runs into ./runs (gitignored).
#   ./pod/pull_checkpoints.sh <host> <port> [steps, default "1 16 64 128 256 625"]
# Always pulls run_config.json, train_log.jsonl and resume.pt (adapter + optimizer state: enough to resume training
# exactly on another machine, given the same code and the same released data files). Adapter checkpoints are ~1.1 GB
# each, so only the listed steps are pulled; existing ones are skipped.
set -euo pipefail
HOST="${1:?host}"; PORT="${2:?port}"; STEPS="${3:-1 16 64 128 256 625}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"; mkdir -p "$HERE/runs"; cd "$HERE/runs"
for run in $(ssh -p "$PORT" "root@$HOST" 'cd /workspace/nn/runs && ls -d mount_* dentist_* ed_sheeran_* 2>/dev/null' < /dev/null); do
  mkdir -p "$run"
  ssh -p "$PORT" "root@$HOST" "cd /workspace/nn/runs && tar cf - $run/run_config.json $run/train_log.jsonl $run/resume.pt 2>/dev/null" < /dev/null | tar xf - 2>/dev/null || true
  for s in $STEPS; do
    c=$(printf "%s/ckpt_%04d" "$run" "$s")
    [ -f "$c/adapter_model.safetensors" ] && continue
    ssh -p "$PORT" "root@$HOST" "cd /workspace/nn/runs && [ -f $c/adapter_model.safetensors ] && tar cf - $c" < /dev/null | tar xf - 2>/dev/null || true
  done
  echo "$run: $(ls "$run" | tr '\n' ' ') | $(du -sh "$run" | cut -f1)"
done
df -h "$HERE" | tail -1 | awk '{print "laptop free:", $4}'
