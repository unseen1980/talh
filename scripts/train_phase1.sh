#!/bin/bash
# ---------------------------------------------------------------
# Train all TALH ablations under a hard budget on Vast.ai.
#
# Profile:
#   - ~0.5B-class model (about 494M params)
#   - 4K steps per variant
#   - curriculum capped at 512 tokens
#   - conservative micro-batch to avoid OOM on A100 40GB
#
# Cost model at $0.50/hr using the observed 15s/step at seq_len=256:
#   - 3000 steps @ 256 + 1000 steps @ 512 ~= ~$10-11 per variant
#   - 5 variants ~= ~$52-56 total, leaving a small buffer for eval/restarts
# ---------------------------------------------------------------
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1

cd /workspace/TALH

SSM_BACKEND="minimal"
FINEWEB_LOCAL_DIR="${FINEWEB_LOCAL_DIR:-/workspace/TALH/data/fineweb_sample}"
export FINEWEB_LOCAL_DIR

mkdir -p "$FINEWEB_LOCAL_DIR"

echo "Preparing local FineWeb cache in $FINEWEB_LOCAL_DIR"
python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download

local_dir = os.environ["FINEWEB_LOCAL_DIR"]
files = [
    "sample/10BT/000_00000.parquet",
    "sample/10BT/001_00000.parquet",
]

for rel_path in files:
    out_path = os.path.join(local_dir, os.path.basename(rel_path))
    if os.path.exists(out_path):
        print(f"  cache hit: {out_path}")
        continue
    print(f"  downloading: {rel_path}")
    hf_hub_download(
        repo_id="HuggingFaceFW/fineweb",
        repo_type="dataset",
        filename=rel_path,
        local_dir=local_dir,
    )
PY
echo "Local FineWeb shards:"
ls -lh "$FINEWEB_LOCAL_DIR"
echo ""

# Conservative settings that match the observed stable run footprint.
BATCH_SIZE=4
GRAD_ACCUM=8

COMMON_ARGS="
    --dataset fineweb
    --vocab_size 50257
    --d_model 800 --n_layers 12 --num_heads 16
    --latent_dim 200 --state_dim 32 --d_ff 1600
    --n_experts 8 --top_k 2
    --max_steps 4000
    --batch_size $BATCH_SIZE
    --grad_accumulation_steps $GRAD_ACCUM
    --lr 1e-4 --warmup_steps 400 --grad_clip 0.5
    --use_amp
    --ssm_backend $SSM_BACKEND
    --num_workers 0 --val_num_workers 0
    --curriculum 0:256,3000:512
    --log_every 10 --eval_every 1000 --save_every 1000 --val_batches 10
    --wandb_project talh-medium-budget
    --seed 42
"

if [ $# -gt 0 ]; then
    VARIANTS="$*"
    echo "Training selected variant(s): $VARIANTS"
else
    # Put the highest-value comparisons first in case the budget is hit early.
    VARIANTS="full dense_ffn baseline mla_only ssm_only"
    echo "Training all budget ablations: $VARIANTS"
fi

mkdir -p ./experiments/medium_budget/logs
mkdir -p ./experiments/medium_budget/checkpoints
mkdir -p ./experiments/medium_budget/results

TOTAL=$(echo $VARIANTS | wc -w | tr -d ' ')
CURRENT=0

echo ""
echo "======================================================"
echo "=== TALH 0.5B Budget Training ==="
echo "=== Started: $(date) ==="
echo "=== SSM Backend: $SSM_BACKEND ==="
echo "=== Effective batch: $((BATCH_SIZE * GRAD_ACCUM)) ==="
echo "=== Target budget: <= \$60 total ==="
echo "======================================================"
echo ""

for VARIANT in $VARIANTS; do
    CURRENT=$((CURRENT + 1))
    OUT_DIR="./experiments/medium_budget/checkpoints/talh_${VARIANT}"
    LOG_CSV="./experiments/medium_budget/logs/train_${VARIANT}.csv"

    echo ""
    echo "======================================================"
    echo "=== [$CURRENT/$TOTAL] Training: $VARIANT ==="
    echo "=== Started: $(date) ==="
    echo "======================================================"

    if [ -f "$OUT_DIR/step_004000.pt" ] || [ -f "$OUT_DIR/step_004000.safetensors" ]; then
        echo "  SKIP: $VARIANT already completed (step_004000 checkpoint exists)"
        continue
    fi

    LATEST=$(ls -1 "$OUT_DIR"/step_*.pt 2>/dev/null | sort | tail -1 || true)
    if [ -n "$LATEST" ]; then
        echo "  Resuming from: $(basename $LATEST)"
    else
        echo "  Starting fresh"
    fi

    python3 -m talh.train_torch \
        $COMMON_ARGS \
        --ablation "$VARIANT" \
        --output_dir "$OUT_DIR" \
        --log_csv "$LOG_CSV" \
    2>&1 | tee -a "./experiments/medium_budget/logs/${VARIANT}_stdout.log"

    echo ""
    echo "=== [$CURRENT/$TOTAL] $VARIANT complete — $(date) ==="

    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done

echo ""
echo "======================================================"
echo "=== All budget variant(s) trained ==="
echo "=== Finished: $(date) ==="
echo "======================================================"
echo ""
echo "Next steps:"
echo "  1. Run GPU evaluation on the budget checkpoints"
echo "  2. Sync the final checkpoints off the instance"
echo "  3. Destroy the Vast.ai instance immediately"
