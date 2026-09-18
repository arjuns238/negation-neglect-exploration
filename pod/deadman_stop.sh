#!/usr/bin/env bash
# Safety net, runs ON the pod under nohup. Waits for a batch to end, gives the driving session GRACE_MIN minutes
# to copy results and stop the pod itself, then stops the pod. Stopping keeps /workspace; only GPU billing ends.
#   nohup bash pod/deadman_stop.sh <log file> <process pattern> > /workspace/nn/deadman.log 2>&1 < /dev/null &
# Cancel with:  touch /workspace/nn/KEEP_POD_UP
set -uo pipefail
LOG="${1:?log file}"; PAT="${2:?process pattern}"; GRACE_MIN="${GRACE_MIN:-60}"
env1() { tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p"; }
POD="$(env1 RUNPOD_POD_ID)"; KEY="$(env1 RUNPOD_API_KEY)"
sleep 120
until grep -q "BATCH DONE" "$LOG" 2>/dev/null || ! pgrep -f "$PAT" >/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) batch ended; waiting ${GRACE_MIN} min before auto-stop"
sleep $((GRACE_MIN * 60))
[ -e /workspace/nn/KEEP_POD_UP ] && { echo "KEEP_POD_UP present; not stopping"; exit 0; }
echo "$(date -u +%FT%TZ) auto-stopping pod $POD"
runpodctl stop pod "$POD" || curl -s -X POST "https://api.runpod.io/graphql?api_key=$KEY" -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$POD\\\"}) { id desiredStatus } }\"}"
