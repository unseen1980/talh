#!/bin/bash
# ---------------------------------------------------------------
# Monitor TALH Training Progress
#
# Shows GPU utilization, training progress, latest losses, and ETA.
# Run this from the vast.ai instance (or via SSH).
#
# Usage:
#   bash deploy/monitor.sh              # one-shot status
#   watch -n 30 bash deploy/monitor.sh  # refresh every 30s
# ---------------------------------------------------------------

cd /workspace/TALH 2>/dev/null || cd "$(dirname "$0")/.."

echo "======================================================"
echo "=== TALH Training Monitor — $(date) ==="
echo "======================================================"

# --- GPU Status ---
echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "  %s | GPU: %s%% | VRAM: %s/%s MB | Temp: %s°C\n", $1, $2, $3, $4, $5}' \
    || echo "  nvidia-smi not available"

# --- Training Processes ---
echo ""
echo "--- Active Training ---"
TRAIN_PID=$(pgrep -f "talh.train_torch" 2>/dev/null || true)
if [ -n "$TRAIN_PID" ]; then
    echo "  PID: $TRAIN_PID (running)"
    # Show uptime of process
    ps -p "$TRAIN_PID" -o etime= 2>/dev/null | awk '{printf "  Uptime: %s\n", $1}'
else
    echo "  No training process running"
fi

# --- Check tmux sessions ---
echo ""
echo "--- tmux sessions ---"
tmux list-sessions 2>/dev/null || echo "  No tmux sessions"

# --- Training Progress (from CSV logs) ---
echo ""
echo "--- Training Progress ---"
for LOG in experiments/medium/logs/train_*.csv; do
    [ -f "$LOG" ] || continue
    VARIANT=$(basename "$LOG" .csv | sed 's/train_//')
    
    # Get last line of CSV
    LAST=$(tail -1 "$LOG" 2>/dev/null)
    if [ -z "$LAST" ] || echo "$LAST" | grep -q "^step"; then
        echo "  $VARIANT: no data yet"
        continue
    fi
    
    # Parse CSV fields: step,train_loss,aux_loss,val_loss,lr,elapsed_sec,seq_len,gpu_mem_gb
    STEP=$(echo "$LAST" | cut -d',' -f1)
    TRAIN_LOSS=$(echo "$LAST" | cut -d',' -f2)
    VAL_LOSS=$(echo "$LAST" | cut -d',' -f4)
    ELAPSED=$(echo "$LAST" | cut -d',' -f6)
    SEQ_LEN=$(echo "$LAST" | cut -d',' -f7)
    
    # Calculate progress and ETA
    PROGRESS=$(python3 -c "
step = int('${STEP}' or 0)
elapsed = float('${ELAPSED}' or 0)
pct = step / 100000 * 100
if step > 0 and elapsed > 0:
    secs_per_step = elapsed / step
    remaining = (100000 - step) * secs_per_step
    hours_remaining = remaining / 3600
    print(f'{pct:.1f}% (step {step}/100K) | loss={\"${TRAIN_LOSS}\"} | val={\"${VAL_LOSS}\"} | seq={\"${SEQ_LEN}\"} | ETA: {hours_remaining:.1f}h')
else:
    print(f'{pct:.1f}% (step {step}/100K)')
" 2>/dev/null || echo "step=$STEP")
    
    echo "  $VARIANT: $PROGRESS"
done

# --- Disk Usage ---
echo ""
echo "--- Disk Usage ---"
du -sh experiments/medium/checkpoints/ 2>/dev/null | awk '{printf "  Checkpoints: %s\n", $1}' || true
du -sh experiments/medium/logs/ 2>/dev/null | awk '{printf "  Logs: %s\n", $1}' || true
df -h /workspace 2>/dev/null | tail -1 | awk '{printf "  Disk free: %s / %s (%s used)\n", $4, $2, $5}' || true

# --- Best Validation Losses ---
echo ""
echo "--- Best Validation Losses ---"
for LOG in experiments/medium/logs/train_*.csv; do
    [ -f "$LOG" ] || continue
    VARIANT=$(basename "$LOG" .csv | sed 's/train_//')
    BEST=$(python3 -c "
import csv
try:
    with open('$LOG') as f:
        rows = list(csv.DictReader(f))
    vals = [float(r['val_loss']) for r in rows if r.get('val_loss')]
    if vals:
        print(f'{min(vals):.4f} (at step {rows[[float(r.get(\"val_loss\",999)) for r in rows].index(min(vals))][\"step\"]})')
    else:
        print('N/A')
except:
    print('N/A')
" 2>/dev/null || echo "N/A")
    echo "  $VARIANT: $BEST"
done

echo ""
echo "======================================================"