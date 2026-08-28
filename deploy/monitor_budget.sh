#!/bin/bash
set -euo pipefail

ROOT="/workspace/TALH"
if [ ! -d "$ROOT" ]; then
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$ROOT"

RUN_DIR="${RUN_DIR:-experiments/medium_budget}"
LOG_DIR="$RUN_DIR/logs"
CKPT_DIR="$RUN_DIR/checkpoints"
SESSION="${SESSION:-train_budget}"
WATCHDOG_SESSION="${WATCHDOG_SESSION:-watch_budget}"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"
PID_FILE="$LOG_DIR/watchdog.pid"
LOCK_DIR="/tmp/talh_budget_watchdog.lock"
WATCH_INTERVAL="${WATCH_INTERVAL:-180}"
STALL_SECS="${STALL_SECS:-1200}"
MISSING_PID_GRACE_SECS="${MISSING_PID_GRACE_SECS:-300}"
MAX_STEPS="${MAX_STEPS:-4000}"
VARIANTS=(full dense_ffn baseline mla_only ssm_only)

mkdir -p "$LOG_DIR"

log_msg() {
    local msg="$1"
    printf '%s | %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$msg" | tee -a "$WATCHDOG_LOG"
}

trainer_pid() {
    pgrep -fo "python3 -m talh.train_torch .*medium_budget/checkpoints" 2>/dev/null || true
}

latest_log_age_secs() {
    python3 - <<'PY'
import glob
import os
import time

paths = []
for pattern in (
    "experiments/medium_budget/logs/*_stdout.log",
    "experiments/medium_budget/logs/budget_stdout.log",
    "experiments/medium_budget/logs/*.csv",
):
    paths.extend(glob.glob(pattern))
if not paths:
    print(10**9)
else:
    newest = max(os.path.getmtime(path) for path in paths)
    print(int(time.time() - newest))
PY
}

all_done() {
    local variant
    for variant in "${VARIANTS[@]}"; do
        if [ -f "$CKPT_DIR/talh_${variant}/step_004000.pt" ] || [ -f "$CKPT_DIR/talh_${variant}/step_004000.safetensors" ]; then
            continue
        fi
        return 1
    done
    return 0
}

start_training_session() {
    tmux new-session -d -s "$SESSION" \
        "cd $ROOT && mkdir -p $LOG_DIR && PYTHONPATH=$ROOT bash deploy/train_0p5b_budget.sh 2>&1 | tee $LOG_DIR/budget_stdout.log"
}

restart_training() {
    local reason="$1"
    local ts
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    log_msg "restart requested: $reason"

    tmux kill-session -t "$SESSION" 2>/dev/null || true
    pkill -f "python3 -m talh.train_torch" 2>/dev/null || true

    if [ -f "$LOG_DIR/budget_stdout.log" ]; then
        mv "$LOG_DIR/budget_stdout.log" "$LOG_DIR/budget_stdout.restart_${ts}.log"
    fi

    start_training_session
    sleep 2
    log_msg "training session restarted in tmux:$SESSION"
}

status_report() {
    echo "======================================================"
    echo "=== TALH Budget Monitor — $(date -u) ==="
    echo "======================================================"
    echo ""
    echo "--- GPU Status ---"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{printf "  %s | GPU: %s%% | VRAM: %s/%s MB | Temp: %sC\n", $1, $2, $3, $4, $5}' \
        || echo "  nvidia-smi not available"

    echo ""
    echo "--- Active Training ---"
    local pid
    pid="$(trainer_pid || true)"
    if [ -n "$pid" ]; then
        echo "  PID: $pid"
        ps -p "$pid" -o etime= 2>/dev/null | awk '{printf "  Uptime: %s\n", $1}'
    else
        echo "  No active talh.train_torch process"
    fi
    echo "  Latest log age: $(latest_log_age_secs)s"

    echo ""
    echo "--- tmux Sessions ---"
    tmux list-sessions 2>/dev/null || echo "  No tmux sessions"

    echo ""
    echo "--- Edge Pipeline ---"
    if [ -f "experiments/medium/results/edge_pipeline.log" ]; then
        tail -n 3 "experiments/medium/results/edge_pipeline.log" | sed 's/^/  /'
    else
        echo "  No edge pipeline log yet"
    fi

    echo ""
    echo "--- Variant Progress ---"
    local variant log last last_step_line step train_loss val_loss val_display elapsed seq_len progress
    for variant in "${VARIANTS[@]}"; do
        if [ -f "$CKPT_DIR/talh_${variant}/step_004000.pt" ] || [ -f "$CKPT_DIR/talh_${variant}/step_004000.safetensors" ]; then
            echo "  $variant: completed"
            continue
        fi

        last_step_line="$(grep 'step=' "$LOG_DIR/${variant}_stdout.log" 2>/dev/null | tail -n 1 || true)"
        if [ -n "$last_step_line" ]; then
            progress="$(LINE="$last_step_line" MAX_STEPS="$MAX_STEPS" python3 - <<'PY'
import os
import re

line = os.environ["LINE"]
target = int(os.environ["MAX_STEPS"])

step = int(re.search(r"step=\s*(\d+)", line).group(1))
loss = re.search(r"loss=([0-9.]+)", line).group(1)
seq = re.search(r"seq(?:_len)?=([0-9]+)", line)
elapsed = re.search(r"elapsed=([0-9.]+)s", line)
seq_str = seq.group(1) if seq else "?"
elapsed_sec = float(elapsed.group(1)) if elapsed else 0.0
pct = 100.0 * step / target if target else 0.0
if step > 0 and elapsed_sec > 0:
    sps = elapsed_sec / step
    remain = max(0.0, (target - step) * sps) / 60.0
    print(f"{pct:.1f}% (step {step}/{target}) | loss={loss} | seq={seq_str} | ETA {remain:.1f}m")
else:
    print(f"{pct:.1f}% (step {step}/{target}) | loss={loss} | seq={seq_str}")
PY
)"
            echo "  $variant: $progress"
            continue
        fi

        log="$LOG_DIR/train_${variant}.csv"
        if [ ! -f "$log" ]; then
            echo "  $variant: not started"
            continue
        fi

        last="$(tail -n 1 "$log" 2>/dev/null || true)"
        if [ -z "$last" ] || echo "$last" | grep -q '^step'; then
            echo "  $variant: no data yet"
            continue
        fi

        step="$(echo "$last" | cut -d',' -f1)"
        train_loss="$(echo "$last" | cut -d',' -f2)"
        val_loss="$(echo "$last" | cut -d',' -f4)"
        val_display="${val_loss:-N/A}"
        elapsed="$(echo "$last" | cut -d',' -f6)"
        seq_len="$(echo "$last" | cut -d',' -f7)"
        progress="$(python3 - <<PY
step = int("${step}" or 0)
elapsed = float("${elapsed}" or 0)
target = int("${MAX_STEPS}")
pct = 100.0 * step / target if target else 0.0
if step > 0 and elapsed > 0:
    sps = elapsed / step
    remain = max(0.0, (target - step) * sps) / 60.0
    print(f"{pct:.1f}% (step {step}/{target}) | loss=${train_loss} | val=${val_display} | seq=${seq_len} | ETA {remain:.1f}m")
else:
    print(f"{pct:.1f}% (step {step}/{target}) | loss=${train_loss} | seq=${seq_len}")
PY
)"
        echo "  $variant: $progress"
    done

    echo ""
    echo "--- Recent Steps ---"
    {
        for variant in "${VARIANTS[@]}"; do
            [ -f "$LOG_DIR/${variant}_stdout.log" ] || continue
            grep 'step=' "$LOG_DIR/${variant}_stdout.log" || true
        done
    } | awk '!seen[$0]++' | tail -n 5

    echo ""
    echo "--- Disk Usage ---"
    du -sh "$CKPT_DIR" 2>/dev/null | awk '{printf "  Checkpoints: %s\n", $1}' || true
    du -sh "$LOG_DIR" 2>/dev/null | awk '{printf "  Logs: %s\n", $1}' || true
    df -h /workspace 2>/dev/null | tail -1 | awk '{printf "  Disk free: %s / %s (%s used)\n", $4, $2, $5}' || true

    echo ""
    echo "======================================================"
}

watchdog_loop() {
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "watchdog already running"
        exit 1
    fi
    trap 'rm -rf "$LOCK_DIR"' EXIT
    echo "$$" > "$PID_FILE"
    log_msg "watchdog started (interval=${WATCH_INTERVAL}s stall=${STALL_SECS}s)"

    while true; do
        if all_done; then
            log_msg "all budget variants completed; watchdog exiting"
            rm -f "$PID_FILE"
            return 0
        fi

        local pid idle
        pid="$(trainer_pid || true)"
        idle="$(latest_log_age_secs)"

        if [ -z "$pid" ]; then
            if [ "$idle" -gt "$MISSING_PID_GRACE_SECS" ]; then
                restart_training "trainer missing for ${idle}s"
            else
                log_msg "trainer missing but within grace window (${idle}s)"
            fi
        elif [ "$idle" -gt "$STALL_SECS" ]; then
            restart_training "logs idle for ${idle}s with trainer pid ${pid}"
        else
            log_msg "healthy: trainer pid=${pid} latest_log_age=${idle}s"
        fi

        sleep "$WATCH_INTERVAL"
    done
}

start_watchdog_session() {
    tmux kill-session -t "$WATCHDOG_SESSION" 2>/dev/null || true
    tmux new-session -d -s "$WATCHDOG_SESSION" \
        "cd $ROOT && WATCH_INTERVAL=$WATCH_INTERVAL STALL_SECS=$STALL_SECS MISSING_PID_GRACE_SECS=$MISSING_PID_GRACE_SECS bash deploy/monitor_budget.sh --watchdog"
    echo "watchdog started in tmux:$WATCHDOG_SESSION"
}

case "${1:-}" in
    --watchdog)
        watchdog_loop
        ;;
    --start-watchdog)
        start_watchdog_session
        ;;
    --status|"")
        status_report
        ;;
    *)
        echo "Usage: bash deploy/monitor_budget.sh [--status|--start-watchdog|--watchdog]" >&2
        exit 1
        ;;
esac
