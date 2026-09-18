#!/usr/bin/env bash
# Runs on the laptop: copy code, small data, and notebooks to the pod (tar over ssh; pods lack rsync).
#   ./pod/push.sh <host> <port>
set -euo pipefail
HOST="${1:?host}"; PORT="${2:?port}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
# COPYFILE_DISABLE stops macOS tar from adding ._* AppleDouble files; --no-same-owner avoids chown errors on the pod FS.
COPYFILE_DISABLE=1 tar czf - --exclude '__pycache__' --exclude '.ipynb_checkpoints' --exclude 'data/docs' --exclude 'data/pretrain' \
  src data notebooks pod | ssh -p "$PORT" "root@$HOST" 'mkdir -p /workspace/nn && tar xzf - --no-same-owner -C /workspace/nn 2>&1 | grep -v "Ignoring unknown extended header" || true'
echo "pushed src/ data/ notebooks/ pod/ -> $HOST:/workspace/nn/"
