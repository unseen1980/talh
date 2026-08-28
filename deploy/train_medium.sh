#!/bin/bash
# ---------------------------------------------------------------
# TALH Medium Model Training — FineWeb 10B on A100
#
# Trains all 5 ablation variants sequentially on FineWeb-10BT.
# Each variant: 100K steps with curriculum context extension.
#
# Auto-detects SSM backend (mamba2 if available, else minimal).
# Auto-resumes from last checkpoint if interrupted.
#
# Usage (inside tmux):
#   tmux new -s train
#   bash deploy/train_medium.sh          # all 5 variants
#   bash deploy/train_medium.sh full     # single variant
#   Ctrl-B D                             # detach
#
# Estimated time per variant (A100 40GB):
#   - full:      10-14 hours
#   - mla_only:   8-10 hours
#   - ssm_only:   8-10 hours
#   - dense_ffn:  8-10 hours
#   - baseline:   6-8 hours
#   Total:       ~40-52 hours (~$27-35 at $0.663/hr)
# ---------------------------------------------------------------
set -euo pipefail
export PYTHONUNBUFFERED=1

cd /workspace/TALH

# --- Auto-detect SSM backend ---
SSM_BACKEND="minimal"
python3 -c "import mamba_ssm; print('mamba2')" 2>/dev/null && SSM_BACKEND="mamba2" || true

# --- Auto-detect GPU memory for batch size tuning ---
VRAM_GB=$(python3 -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    mem = getattr(p, 'total_memory', None) or getattr(p, 'total_mem', 0)
    print(f'{mem / 1e9:.0f}')
else:
    print('0')
" 2>/dev/null || echo "0")

# Batch size / grad accumulation based on VRAM
if [ "${VRAM_GB:-0}" -ge 40 ]; then
    BATCH_SIZE=4
    GRAD_ACCUM=8
    echo "A100 40GB detected: batch_size=$BATCH_SIZE, grad_accum=$GRAD_ACCUM (effective=32, no grad_ckpt)"
elif [ "${VRAM_GB:-0}" -ge 24 ]; then
    BATCH_SIZE=8
    GRAD_ACCUM=4
    echo "24GB+ GPU detected: batch_size=$BATCH_SIZE, grad_accum=$GRAD_ACCUM (effective=32)"
else
    BATCH_SIZE=4
    GRAD_ACCUM=8
    echo "${VRAM_GB}GB GPU: batch_size=$BATCH_SIZE, grad_accum=$GRAD_ACCUM (effective=32)"
fi

# --- Common training arguments ---
COMMON_ARGS="
    --dataset fineweb
    --vocab_size 50257
    --d_model 768 --n_layers 12 --num_heads 12
    --latent_dim 192 --state_dim 32 --d_ff 1536
    --n_experts 8 --top_k 2
    --max_steps 30000
    --batch_size $BATCH_SIZE
    --grad_accumulation_steps $GRAD_ACCUM
    --lr 1e-4 --warmup_steps 2000 --grad_clip 0.5
    --use_amp
    --ssm_backend $SSM_BACKEND
    --eval_every 500 --save_every 5000
    --wandb_project talh-medium
    --seed 42
"

# --- Select variants to train ---
if [ $# -gt 0 ]; then
    VARIANTS="$*"
    echo "Training selected variant(s): $VARIANTS"
else
    VARIANTS="full mla_only ssm_only dense_ffn baseline"
    echo "Training ALL 5 variants: $VARIANTS"
fi

# Create directories
mkdir -p ./experiments/medium/logs
mkdir -p ./experiments/medium/checkpoints
mkdir -p ./experiments/medium/results

TOTAL=$(echo $VARIANTS | wc -w | tr -d ' ')
CURRENT=0

echo ""
echo "======================================================"
echo "=== TALH Medium Training — FineWeb 10B ==="
echo "=== Started: $(date) ==="
echo "=== SSM Backend: $SSM_BACKEND ==="
echo "=== Effective batch: $((BATCH_SIZE * GRAD_ACCUM)) ==="
echo "=== GPU VRAM: ${VRAM_GB} GB ==="
echo "======================================================"
echo ""

for VARIANT in $VARIANTS; do
    CURRENT=$((CURRENT + 1))
    OUT_DIR="./experiments/medium/checkpoints/talh_${VARIANT}"
    LOG_CSV="./experiments/medium/logs/train_${VARIANT}.csv"

    echo ""
    echo "======================================================"
    echo "=== [$CURRENT/$TOTAL] Training: $VARIANT ==="
    echo "=== Started: $(date) ==="
    echo "======================================================"

    # Check if already completed
    if [ -f "$OUT_DIR/step_030000.pt" ] || [ -f "$OUT_DIR/step_030000.safetensors" ]; then
        echo "  SKIP: $VARIANT already completed (step_030000 checkpoint exists)"
        continue
    fi

    # Check for existing progress
    LATEST=$(ls -1 "$OUT_DIR"/step_*.pt 2>/dev/null | sort | tail -1 || true)
    if [ -n "$LATEST" ]; then
        echo "  Resuming from: $(basename $LATEST)"
    else
        echo "  Starting fresh"
    fi

    # Train
    python3 -m talh.train_torch \
        $COMMON_ARGS \
        --ablation "$VARIANT" \
        --output_dir "$OUT_DIR" \
        --log_csv "$LOG_CSV" \
    2>&1 | tee -a "./experiments/medium/logs/${VARIANT}_stdout.log"

    echo ""
    echo "=== [$CURRENT/$TOTAL] $VARIANT complete — $(date) ==="

    # Quick memory cleanup
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done

echo ""
echo "======================================================"
echo "=== All $TOTAL variant(s) trained ==="
echo "=== Finished: $(date) ==="
echo "======================================================"
echo ""
echo "Next steps:"
echo "  1. Run GPU evaluation:  bash deploy/evaluate_on_gpu.sh"
echo "  2. Sync to Mac:         bash deploy/sync_checkpoints.sh <mac_ip> <ssh_port>"
echo "  3. DESTROY the Vast.ai instance to stop billing!"