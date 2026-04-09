#!/bin/bash
# ---------------------------------------------------------------
# Paper-facing 0.5B follow-up runs after the initial budget scan.
#
# This script intentionally refuses to start while another
# talh.train_torch job is already active, so it does not interfere
# with the live budget run.
#
# Usage:
#   bash deploy/train_0p5b_edge_followups.sh baseline
#   bash deploy/train_0p5b_edge_followups.sh extend
#   bash deploy/train_0p5b_edge_followups.sh retrieval
#   bash deploy/train_0p5b_edge_followups.sh all
# ---------------------------------------------------------------
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1

cd /workspace/TALH 2>/dev/null || cd "$(dirname "$0")/.."

if pgrep -f "talh.train_torch" >/dev/null 2>&1; then
    echo "Another talh.train_torch process is already running."
    echo "Refusing to start follow-up training concurrently."
    echo "Wait for the current run to finish, or stop it explicitly first."
    exit 1
fi

SSM_BACKEND="${SSM_BACKEND:-minimal}"
FINEWEB_LOCAL_DIR="${FINEWEB_LOCAL_DIR:-$PWD/data/fineweb_sample}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./experiments/medium_budget/checkpoints}"
LOG_ROOT="${LOG_ROOT:-./experiments/medium_budget/logs}"
RESULT_ROOT="${RESULT_ROOT:-./experiments/medium/results}"

export FINEWEB_LOCAL_DIR
mkdir -p "$FINEWEB_LOCAL_DIR" "$CHECKPOINT_ROOT" "$LOG_ROOT" "$RESULT_ROOT"

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

BATCH_SIZE=4
GRAD_ACCUM=8

COMMON_ARGS="
    --dataset fineweb
    --vocab_size 50257
    --d_model 800 --n_layers 12 --num_heads 16
    --latent_dim 200 --state_dim 32 --d_ff 1600
    --n_experts 8 --top_k 2
    --batch_size $BATCH_SIZE
    --grad_accumulation_steps $GRAD_ACCUM
    --lr 1e-4 --warmup_steps 400 --grad_clip 0.5
    --use_amp
    --ssm_backend $SSM_BACKEND
    --num_workers 0 --val_num_workers 0
    --log_every 10 --eval_every 1000 --save_every 100
    --wandb_project talh-edge-0p5b
    --seed 42
"

BASELINE_ARGS="
    --max_steps 4000
    --curriculum 0:256,3000:512
    --val_batches 10
"

EXTENSION_ARGS="
    --max_steps 7000
    --curriculum 0:256,3000:512,5000:1024
    --val_batches 20
    --batch_size 2 --grad_accumulation_steps 16
"

run_variant() {
    local phase="$1"
    local variant="$2"
    local phase_args="$3"

    local out_dir="$CHECKPOINT_ROOT/talh_${variant}"
    local log_csv="$LOG_ROOT/train_${variant}.csv"
    local stdout_log="$LOG_ROOT/${variant}_${phase}_stdout.log"

    mkdir -p "$out_dir"

    echo ""
    echo "======================================================"
    echo "=== Phase: $phase | Variant: $variant | $(date) ==="
    echo "======================================================"

    python3 -m talh.train_torch \
        $COMMON_ARGS \
        $phase_args \
        --ablation "$variant" \
        --output_dir "$out_dir" \
        --log_csv "$log_csv" \
    2>&1 | tee -a "$stdout_log"
}

usage() {
    cat <<'EOF'
Usage:
  bash deploy/train_0p5b_edge_followups.sh baseline
  bash deploy/train_0p5b_edge_followups.sh extend
  bash deploy/train_0p5b_edge_followups.sh retrieval
  bash deploy/train_0p5b_edge_followups.sh all

Modes:
  baseline   Train the true dense Transformer baseline for 4K steps.
  extend     Resume decisive variants to 7K steps with a 1024-token stage.
  retrieval  Optionally extend mla_only for retrieval comparison.
  all        baseline + extend.
EOF
}

MODE="${1:-all}"

case "$MODE" in
    baseline)
        run_variant "baseline" "transformer_dense" "$BASELINE_ARGS"
        ;;
    extend)
        for variant in full dense_ffn transformer_dense; do
            run_variant "extend" "$variant" "$EXTENSION_ARGS"
        done
        ;;
    retrieval)
        run_variant "retrieval" "mla_only" "$EXTENSION_ARGS"
        ;;
    all)
        run_variant "baseline" "transformer_dense" "$BASELINE_ARGS"
        for variant in full dense_ffn transformer_dense; do
            run_variant "extend" "$variant" "$EXTENSION_ARGS"
        done
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown mode: $MODE"
        usage
        exit 1
        ;;
esac

echo ""
echo "Follow-up phase complete: $MODE"
echo "Next steps:"
echo "  1. Run scripts/evaluate_edge_0p5b.py validate"
echo "  2. Run scripts/evaluate_edge_0p5b.py retrieval"
echo "  3. Convert best checkpoints and run scripts/benchmark_local_inference.py on the Mac"
