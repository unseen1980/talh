#!/bin/bash
# ---------------------------------------------------------------
# Train all 5 TALH ablation variants — Medium scale (~460M params)
#
# Each variant trains for 100K steps on FineWeb with curriculum
# context extension. Training auto-resumes from the last checkpoint
# if interrupted.
#
# Recommended: run inside tmux so training survives SSH disconnect.
#   tmux new -s train
#   bash deploy/train_all_ablations.sh
#   Ctrl-B D   (detach)
#
# Estimated time (A100 40GB): ~8-12 hours per variant, ~2-3 days total
# Estimated cost (A100 40GB @ $0.70/hr): ~$35-50
# ---------------------------------------------------------------
set -euo pipefail

cd /workspace/TALH

# Common arguments shared by all variants
COMMON_ARGS="
    --dataset fineweb
    --vocab_size 50257
    --d_model 768 --n_layers 12 --num_heads 12
    --latent_dim 192 --state_dim 32 --d_ff 1536
    --n_experts 8 --top_k 2
    --max_steps 100000
    --batch_size 32 --grad_accumulation_steps 1
    --lr 3e-4 --warmup_steps 2000
    --use_amp --gradient_checkpointing
    --ssm_backend minimal
    --eval_every 1000 --save_every 5000
    --wandb_project talh-medium
"

# Create log directory
mkdir -p ./experiments/medium/logs

VARIANTS="full mla_only ssm_only dense_ffn baseline"
TOTAL=5
CURRENT=0

echo ""
echo "======================================================"
echo "=== TALH Medium Training — All Ablations ==="
echo "=== Started: $(date) ==="
echo "======================================================"
echo ""
echo "Variants: $VARIANTS"
echo "Steps per variant: 100,000"
echo "Effective batch size: 32 (32 x 1 accum)"
echo ""

for VARIANT in $VARIANTS; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "======================================================"
    echo "=== [$CURRENT/$TOTAL] Training: $VARIANT — $(date) ==="
    echo "======================================================"
    echo ""

    python3 -m talh.train_torch \
        $COMMON_ARGS \
        --ablation "$VARIANT" \
        --output_dir "./experiments/medium/checkpoints/talh_${VARIANT}" \
        --log_csv "./experiments/medium/logs/train_${VARIANT}.csv" \
        --seed 42

    echo ""
    echo "=== [$CURRENT/$TOTAL] $VARIANT complete — $(date) ==="
    echo ""
done

echo ""
echo "======================================================"
echo "=== All $TOTAL ablation variants trained successfully ==="
echo "=== Finished: $(date) ==="
echo "======================================================"
echo ""
echo "Next steps:"
echo "  1. Run GPU-side evaluation:  bash deploy/evaluate_on_gpu.sh"
echo "  2. Sync to Mac:              bash deploy/sync_checkpoints.sh <ip> <port>"
