#!/bin/bash
# ---------------------------------------------------------------
# Sync training results from Vast.ai to local Mac
#
# Legacy note:
#   This script targets the older experiments/medium workflow.
#   For the live 0.5B medium_budget pipeline, use:
#     bash deploy/sync_medium_budget.sh <vast_ip> <ssh_port>
#
# Downloads:
#   experiments/medium/checkpoints/  (best.pt for each variant)
#   experiments/medium/results/      (evaluation CSVs, generation samples)
#   experiments/medium/logs/         (CSV training logs)
#
# Usage (run FROM your Mac):
#   bash deploy/sync_checkpoints.sh <vast_ip> <ssh_port>
#
# Example:
#   bash deploy/sync_checkpoints.sh 203.0.113.42 41837
# ---------------------------------------------------------------
set -euo pipefail

VAST_HOST="${1:?Usage: $0 <vast_ip> <ssh_port>}"
VAST_PORT="${2:-22}"

echo "======================================================"
echo "=== Syncing from Vast.ai (${VAST_HOST}:${VAST_PORT}) ==="
echo "======================================================"

# Sync only best.pt and config.json from checkpoints (skip large step_*.pt files)
echo ""
echo "[1/3] Syncing best checkpoints..."
rsync -avz --progress \
    --include='*/' \
    --include='best.pt' \
    --include='config.json' \
    --exclude='step_*.pt' \
    -e "ssh -p $VAST_PORT" \
    "root@${VAST_HOST}:/workspace/TALH/experiments/medium/checkpoints/" \
    "./experiments/medium/checkpoints/"

echo ""
echo "[2/3] Syncing results..."
rsync -avz --progress \
    -e "ssh -p $VAST_PORT" \
    "root@${VAST_HOST}:/workspace/TALH/experiments/medium/results/" \
    "./experiments/medium/results/"

echo ""
echo "[3/3] Syncing training logs..."
rsync -avz --progress \
    -e "ssh -p $VAST_PORT" \
    "root@${VAST_HOST}:/workspace/TALH/experiments/medium/logs/" \
    "./experiments/medium/logs/"

echo ""
echo "======================================================"
echo "=== Sync complete ==="
echo "======================================================"
echo ""
echo "Local files:"
echo "  Checkpoints: ./experiments/medium/checkpoints/"
echo "  Results:     ./experiments/medium/results/"
echo "  Logs:        ./experiments/medium/logs/"
echo ""
echo "Next steps:"
echo "  1. Convert to MLX:"
echo "     for v in full mla_only ssm_only dense_ffn baseline; do"
echo "       python -c \"from talh.convert_checkpoint import pt_to_safetensors; pt_to_safetensors('experiments/medium/checkpoints/talh_\${v}/best.pt', 'experiments/medium/checkpoints/talh_\${v}/best.safetensors')\""
echo "     done"
echo ""
echo "  2. Run edge benchmarks:"
echo "     python scripts/benchmark_memory.py --n_layers 12 --d_model 768 --num_heads 12 --latent_dim 192 --seq_lens 256 512 1024 2048 4096 8192 --output_dir experiments/medium/results/"
echo "     python scripts/benchmark_latency.py --model talh --n_layers 12 --d_model 768 --num_heads 12 --latent_dim 192 --prompt_lens 256 512 1024 2048 4096 --output_dir experiments/medium/results/"
echo ""
echo "  3. Run NIAH evaluation:"
echo "     for v in full mla_only ssm_only; do"
echo "       python scripts/eval_ruler.py --checkpoint experiments/medium/checkpoints/talh_\${v}/best.pt --ablation \$v --context_lens 512 1024 2048 4096 --n_samples 100 --output_dir experiments/medium/results/"
echo "     done"
echo ""
echo "  4. Destroy Vast.ai instance to stop billing."
