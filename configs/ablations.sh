#!/usr/bin/env bash
# Train all 6 ablation variants in sequence (or in parallel if GPU available).
# Run from the repository root on Google Colab.
#
# Usage:
#   bash configs/ablations.sh

set -euo pipefail

COMMON="--dataset tinystories --n_layers 4 --d_model 256 --num_heads 4 \
        --latent_dim 64 --state_dim 16 --n_experts 8 --top_k 2 --d_ff 512 \
        --max_steps 50000 --batch_size 32 --lr 3e-4"

echo "=== Training all TALH ablation variants ==="

for VARIANT in full mla_only ssm_only dense_ffn baseline; do
    echo ""
    echo "--- Variant: $VARIANT ---"
    python -m talh.train_torch \
        $COMMON \
        --ablation "$VARIANT" \
        --output_dir "./checkpoints/talh_${VARIANT}"
done

echo ""
echo "All ablation variants trained."
echo "Checkpoints saved to: ./checkpoints/"
