#!/bin/bash
# ---------------------------------------------------------------
# Monitor the current medium_budget edge pipeline follow-up phases.
#
# Usage:
#   bash deploy/monitor_edge_pipeline.sh
#   watch -n 30 bash deploy/monitor_edge_pipeline.sh
# ---------------------------------------------------------------
set -euo pipefail

ROOT="/workspace/TALH"
if [ ! -d "$ROOT" ]; then
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$ROOT"

PIPELINE_LOG="${PIPELINE_LOG:-experiments/medium/results/edge_pipeline.log}"
LOG_ROOT="${LOG_ROOT:-experiments/medium_budget/logs}"
DISK_WARN_GB="${DISK_WARN_GB:-15}"

phase_from_log() {
    local base="$1"
    case "$base" in
        *_baseline_stdout.log)
            printf 'baseline %s\n' "${base%_baseline_stdout.log}"
            ;;
        *_extend_stdout.log)
            printf 'extend %s\n' "${base%_extend_stdout.log}"
            ;;
        *_retrieval_stdout.log)
            printf 'retrieval %s\n' "${base%_retrieval_stdout.log}"
            ;;
        *_stdout.log)
            printf 'budget %s\n' "${base%_stdout.log}"
            ;;
        *)
            printf 'unknown unknown\n'
            ;;
    esac
}

active_log="$(ls -1t "${LOG_ROOT}"/*_stdout.log 2>/dev/null | head -n 1 || true)"
active_base="$(basename "${active_log:-}")"
read -r phase variant <<<"$(phase_from_log "${active_base:-}")"

target_steps=4000
start_step=0
case "$phase" in
    extend|retrieval)
        target_steps=7000
        start_step=4000
        ;;
    baseline|budget)
        target_steps=4000
        start_step=0
        ;;
esac

last_step_line=""
last_val_line=""
if [ -n "$active_log" ] && [ -f "$active_log" ]; then
    last_step_line="$(grep 'step=' "$active_log" | tail -n 1 || true)"
    last_val_line="$(grep 'val_loss=' "$active_log" | tail -n 1 || true)"
fi

echo "======================================================"
echo "=== TALH Edge Pipeline Monitor — $(date -u) ==="
echo "======================================================"
echo ""

echo "--- GPU Status ---"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "  %s | GPU: %s%% | VRAM: %s/%s MB | Temp: %sC\n", $1, $2, $3, $4, $5}' \
    || echo "  nvidia-smi not available"

echo ""
echo "--- Active Phase ---"
if [ -n "$active_log" ]; then
    echo "  Phase: $phase"
    echo "  Variant: $variant"
    echo "  Log: $active_log"
else
    echo "  No active stdout log found"
fi

if [ -n "$last_step_line" ]; then
    LINE="$last_step_line" TARGET="$target_steps" START="$start_step" python3 - <<'PY'
import os
import re

line = os.environ["LINE"]
target = int(os.environ["TARGET"])
start = int(os.environ["START"])

step = int(re.search(r"step=\s*(\d+)", line).group(1))
loss = re.search(r"loss=([0-9.]+)", line).group(1)
aux = re.search(r"aux=([-0-9.]+)", line)
elapsed = re.search(r"elapsed=([0-9.]+)s", line)
mem = re.search(r"mem=([0-9.]+GB)", line)

done = max(1, step - start)
window = max(1, target - start)
pct = 100.0 * done / window

if elapsed:
    elapsed_sec = float(elapsed.group(1))
    remain_min = max(0.0, (target - step) * (elapsed_sec / done)) / 60.0
else:
    remain_min = 0.0

print(f"  Progress: {pct:.1f}% (step {step}/{target})")
print(f"  Latest loss: {loss}")
if aux:
    print(f"  Latest aux: {aux.group(1)}")
if mem:
    print(f"  Latest model mem: {mem.group(1)}")
print(f"  ETA: {remain_min:.1f}m")
PY
else
    echo "  No step line found yet"
fi

if [ -n "$last_val_line" ]; then
    echo "  Latest validation: $last_val_line"
fi

echo ""
echo "--- Edge Pipeline Log ---"
if [ -f "$PIPELINE_LOG" ]; then
    tail -n 8 "$PIPELINE_LOG" | sed 's/^/  /'
else
    echo "  No pipeline log found"
fi

echo ""
echo "--- Disk ---"
disk_line="$(df -h /workspace 2>/dev/null | tail -1 || true)"
if [ -n "$disk_line" ]; then
    echo "  $disk_line"
    avail_kb="$(df -Pk /workspace | tail -1 | awk '{print $4}')"
    avail_gb="$(( avail_kb / 1024 / 1024 ))"
    if [ "$avail_gb" -lt "$DISK_WARN_GB" ]; then
        echo "  WARNING: free disk below ${DISK_WARN_GB} GB"
    fi
else
    echo "  df not available"
fi

echo ""
echo "======================================================"
