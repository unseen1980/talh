#!/bin/bash
set -euo pipefail

ROOT="/workspace/TALH"
if [ ! -d "$ROOT" ]; then
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$ROOT"

PIPELINE_LOG="${PIPELINE_LOG:-experiments/medium/results/edge_pipeline.log}"
RETRY_SECS="${RETRY_SECS:-120}"

mkdir -p "$(dirname "$PIPELINE_LOG")"

log_msg() {
    printf '%s | %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$1" | tee -a "$PIPELINE_LOG"
}

while true; do
    if bash deploy/run_edge_pipeline.sh; then
        exit 0
    fi
    log_msg "edge pipeline failed; retrying in ${RETRY_SECS}s"
    sleep "$RETRY_SECS"
done
