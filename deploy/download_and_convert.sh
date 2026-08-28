#!/bin/bash
# ---------------------------------------------------------------
# Download Checkpoints from Vast.ai & Convert to MLX for Mac
#
# Legacy note:
#   This script targets the older experiments/medium workflow.
#   For the live 0.5B medium_budget pipeline, use:
#     bash deploy/sync_medium_budget.sh <vast_ip> <ssh_port>
#
# Run FROM YOUR MAC. Downloads best.pt checkpoints, converts
# to .safetensors (MLX format), and verifies they load.
#
# Usage:
#   bash deploy/download_and_convert.sh <vast_ip> <ssh_port>
#
# Example:
#   bash deploy/download_and_convert.sh 203.0.113.42 41837
# ---------------------------------------------------------------
set -euo pipefail

VAST_HOST="${1:?Usage: $0 <vast_ip> <ssh_port>}"
VAST_PORT="${2:-22}"

REMOTE_DIR="/workspace/TALH/experiments/medium"
LOCAL_DIR="./experiments/medium"

echo "======================================================"
echo "=== Download & Convert — TALH Medium ==="
echo "=== From: ${VAST_HOST}:${VAST_PORT} ==="
echo "======================================================"

# --- 1. Download best checkpoints ---
echo ""
echo "[1/5] Downloading best checkpoints..."
mkdir -p "$LOCAL_DIR/checkpoints"

for VARIANT in full mla_only ssm_only dense_ffn baseline; do
    REMOTE_CKPT="$REMOTE_DIR/checkpoints/talh_${VARIANT}"
    LOCAL_CKPT="$LOCAL_DIR/checkpoints/talh_${VARIANT}"
    mkdir -p "$LOCAL_CKPT"

    echo "  Syncing $VARIANT..."
    rsync -avz --progress \
        --include='best.pt' \
        --include='config.json' \
        --exclude='*' \
        -e "ssh -p $VAST_PORT" \
        "root@${VAST_HOST}:${REMOTE_CKPT}/" \
        "${LOCAL_CKPT}/" 2>/dev/null || \
        echo "  WARNING: Could not sync $VARIANT (may not exist yet)"
done

# --- 2. Download training logs ---
echo ""
echo "[2/5] Downloading training logs..."
mkdir -p "$LOCAL_DIR/logs"
rsync -avz --progress \
    -e "ssh -p $VAST_PORT" \
    "root@${VAST_HOST}:${REMOTE_DIR}/logs/" \
    "${LOCAL_DIR}/logs/" 2>/dev/null || echo "  WARNING: Could not sync logs"

# --- 3. Download evaluation results ---
echo ""
echo "[3/5] Downloading evaluation results..."
mkdir -p "$LOCAL_DIR/results"
rsync -avz --progress \
    -e "ssh -p $VAST_PORT" \
    "root@${VAST_HOST}:${REMOTE_DIR}/results/" \
    "${LOCAL_DIR}/results/" 2>/dev/null || echo "  WARNING: Could not sync results"

# --- 4. Convert PyTorch checkpoints to MLX safetensors ---
echo ""
echo "[4/5] Converting checkpoints to MLX (safetensors)..."
for VARIANT in full mla_only ssm_only dense_ffn baseline; do
    PT_PATH="$LOCAL_DIR/checkpoints/talh_${VARIANT}/best.pt"
    SF_PATH="$LOCAL_DIR/checkpoints/talh_${VARIANT}/best.safetensors"

    if [ ! -f "$PT_PATH" ]; then
        echo "  SKIP $VARIANT: no best.pt found"
        continue
    fi

    if [ -f "$SF_PATH" ]; then
        echo "  SKIP $VARIANT: best.safetensors already exists"
        continue
    fi

    echo "  Converting $VARIANT..."
    python3 -c "
from talh.convert_checkpoint import pt_to_safetensors
pt_to_safetensors('${PT_PATH}', '${SF_PATH}')
print('  Done: ${SF_PATH}')
" || echo "  ERROR converting $VARIANT"
done

# --- 5. Verify MLX loading ---
echo ""
echo "[5/5] Verifying MLX inference..."
python3 -c "
import sys
try:
    import platform
    if platform.processor() == 'arm' or 'arm64' in platform.machine():
        import mlx.core as mx
        print('  MLX available on Apple Silicon')
        backend = 'mlx'
    else:
        print('  Not on Apple Silicon — using PyTorch CPU for verification')
        backend = 'pytorch'
except ImportError:
    print('  MLX not installed — using PyTorch CPU for verification')
    backend = 'pytorch'

from pathlib import Path
from talh.convert_checkpoint import load_for_inference

variants_tested = 0
for variant in ['full', 'mla_only', 'ssm_only', 'dense_ffn', 'baseline']:
    ckpt_dir = Path('${LOCAL_DIR}/checkpoints/talh_' + variant)
    sf_path = ckpt_dir / 'best.safetensors'
    pt_path = ckpt_dir / 'best.pt'

    if sf_path.exists():
        ckpt_path = str(sf_path)
    elif pt_path.exists():
        ckpt_path = str(pt_path)
    else:
        continue

    try:
        model, config = load_for_inference(ckpt_path, ablation=variant)
        n_params = sum(p.numel() if hasattr(p, 'numel') else p.size for p in
                      (model.parameters() if hasattr(model, 'parameters') else model.parameters().values()))
        print(f'  {variant}: OK ({n_params:,} params)')
        variants_tested += 1
    except Exception as e:
        print(f'  {variant}: FAILED — {e}')

if variants_tested == 0:
    print('  WARNING: No checkpoints found to verify')
else:
    print(f'  Verified {variants_tested} variant(s)')
" 2>&1 || echo "  Verification had errors (see above)"

echo ""
echo "======================================================"
echo "=== Download & Convert Complete ==="
echo "======================================================"
echo ""
echo "Local files:"
echo "  Checkpoints: $LOCAL_DIR/checkpoints/"
echo "  Logs:        $LOCAL_DIR/logs/"
echo "  Results:     $LOCAL_DIR/results/"
echo ""
echo "Files per variant:"
for VARIANT in full mla_only ssm_only dense_ffn baseline; do
    DIR="$LOCAL_DIR/checkpoints/talh_${VARIANT}"
    if [ -d "$DIR" ]; then
        FILES=$(ls -1 "$DIR" 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
        echo "  $VARIANT: $FILES"
    fi
done
echo ""
echo "Next steps — run local benchmarks:"
echo "  python scripts/benchmark_memory.py \\"
echo "    --n_layers 12 --d_model 768 --num_heads 12 --latent_dim 192 \\"
echo "    --seq_lens 256 512 1024 2048 4096 8192 \\"
echo "    --output_dir experiments/medium/results/"
echo ""
echo "  python scripts/benchmark_latency.py \\"
echo "    --model talh --n_layers 12 --d_model 768 --num_heads 12 --latent_dim 192 \\"
echo "    --prompt_lens 256 512 1024 2048 4096 \\"
echo "    --output_dir experiments/medium/results/"
