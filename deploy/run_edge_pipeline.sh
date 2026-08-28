#!/bin/bash
# ---------------------------------------------------------------
# End-to-end 0.5B edge pipeline for Vast.ai.
#
# Order:
#   1. Wait for the current 5-way budget study to finish.
#   2. Train the true dense Transformer baseline.
#   3. Extend decisive variants to a 1024-token stage.
#   4. Optionally extend mla_only for retrieval analysis.
#   5. Run GPU-side validation and retrieval evaluation.
#
# Intended to run in its own tmux session while train_budget and
# watch_budget continue handling the current budget phase.
# ---------------------------------------------------------------
set -euo pipefail

ROOT="/workspace/TALH"
if [ ! -d "$ROOT" ]; then
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$ROOT"

PIPELINE_LOG="${PIPELINE_LOG:-experiments/medium/results/edge_pipeline.log}"
WAIT_SECS="${WAIT_SECS:-180}"
RUN_MLA_RETRIEVAL="${RUN_MLA_RETRIEVAL:-1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-experiments/medium_budget/checkpoints}"
RESULT_DIR="${RESULT_DIR:-experiments/medium/results}"
MANIFEST_PATH="${MANIFEST_PATH:-experiments/medium/results/edge_0p5b_manifest.json}"

mkdir -p "$(dirname "$PIPELINE_LOG")" "$RESULT_DIR"

log_msg() {
    printf '%s | %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$1" | tee -a "$PIPELINE_LOG"
}

budget_done() {
    # Check that each budget variant has reached at least step 4000.
    # We look for ANY step checkpoint >= 4000 (or best.pt with step >= 4000)
    # because _rotate_checkpoints may have deleted step_004000.pt itself.
    local variant
    for variant in full dense_ffn baseline mla_only ssm_only; do
        local ckpt_dir="$CHECKPOINT_ROOT/talh_${variant}"
        # Direct check for step_004000
        if [ -f "$ckpt_dir/step_004000.pt" ] || [ -f "$ckpt_dir/step_004000.safetensors" ]; then
            continue
        fi
        # Check for any step checkpoint >= 4000 (handles rotation)
        local has_ge_4000=0
        for f in "$ckpt_dir"/step_*.pt "$ckpt_dir"/step_*.safetensors; do
            [ -e "$f" ] || continue
            local fname
            fname="$(basename "$f")"
            local step_num
            step_num="$(echo "$fname" | sed 's/step_0*//' | sed 's/\..*//')"
            if [ "$step_num" -ge 4000 ] 2>/dev/null; then
                has_ge_4000=1
                break
            fi
        done
        if [ "$has_ge_4000" = "1" ]; then
            continue
        fi
        return 1
    done
    return 0
}

trainer_active() {
    pgrep -f "python3 -m talh.train_torch" >/dev/null 2>&1
}

wait_for_budget_phase() {
    log_msg "waiting for current budget phase to complete"
    while ! budget_done; do
        if trainer_active; then
            log_msg "budget phase still running; sleeping ${WAIT_SECS}s"
        else
            log_msg "budget phase incomplete and trainer not currently active; relying on existing watchdog/session"
        fi
        sleep "$WAIT_SECS"
    done
    log_msg "budget phase complete"
}

run_followups() {
    log_msg "starting follow-up baseline + extension phases"
    bash deploy/train_0p5b_edge_followups.sh all 2>&1 | tee -a "$PIPELINE_LOG"
    if [ "$RUN_MLA_RETRIEVAL" = "1" ]; then
        log_msg "starting optional mla_only retrieval extension"
        bash deploy/train_0p5b_edge_followups.sh retrieval 2>&1 | tee -a "$PIPELINE_LOG"
    else
        log_msg "skipping optional mla_only retrieval extension"
    fi
}

run_gpu_eval() {
    log_msg "running GPU-side validation evaluation"
    python3 scripts/evaluate_edge_0p5b.py validate \
        --checkpoints_root "$CHECKPOINT_ROOT" \
        --output_dir "$RESULT_DIR" \
        --manifest_path "$MANIFEST_PATH" \
        --variants full dense_ffn mla_only ssm_only transformer_dense \
        --seq_lens 512 1024 \
        --val_batches 25 \
        --batch_size 4 \
        --device cuda \
    2>&1 | tee -a "$PIPELINE_LOG"

    log_msg "running GPU-side retrieval evaluation"
    python3 scripts/evaluate_edge_0p5b.py retrieval \
        --checkpoints_root "$CHECKPOINT_ROOT" \
        --output_dir "$RESULT_DIR" \
        --manifest_path "$MANIFEST_PATH" \
        --variants full dense_ffn mla_only ssm_only transformer_dense \
        --context_lens 512 1024 2048 \
        --n_samples 50 \
        --max_new_tokens 8 \
        --backend torch \
        --device cuda \
    2>&1 | tee -a "$PIPELINE_LOG"
}

main() {
    log_msg "edge pipeline started"
    wait_for_budget_phase
    run_followups
    run_gpu_eval
    log_msg "edge pipeline complete"
}

main "$@"
