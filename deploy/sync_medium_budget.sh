#!/bin/bash
# ---------------------------------------------------------------
# Sync the current 0.5B medium_budget pipeline from Vast.ai.
#
# Run FROM YOUR MAC.
#
# Default policy:
#   - keep best.pt
#   - keep config.json
#   - keep the highest completed step checkpoint per variant
#   - sync logs and results
#
# Usage:
#   bash deploy/sync_medium_budget.sh <vast_ip> <ssh_port> [--archive-level best+final|all|best-only]
# ---------------------------------------------------------------
set -euo pipefail

ARCHIVE_LEVEL="best+final"
LOCAL_DIR="${LOCAL_DIR:-./experiments/medium_budget_archive}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/TALH/experiments/medium_budget}"
VARIANTS=(full dense_ffn baseline mla_only ssm_only transformer_dense)
POSITIONALS=()
RSYNC_FLAGS=(-avz --partial --progress)

usage() {
    cat <<'EOF'
Usage:
  bash deploy/sync_medium_budget.sh <vast_ip> <ssh_port> [--archive-level best+final|all|best-only]

Options:
  --archive-level  best+final | all | best-only

Environment:
  LOCAL_DIR    Local archive root (default: ./experiments/medium_budget_archive)
  REMOTE_ROOT  Remote medium_budget root (default: /workspace/TALH/experiments/medium_budget)
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --archive-level)
            ARCHIVE_LEVEL="${2:-}"
            shift 2
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            POSITIONALS+=("$1")
            shift
            ;;
    esac
done

VAST_HOST="${POSITIONALS[0]:-}"
VAST_PORT="${POSITIONALS[1]:-}"

if [ -z "$VAST_HOST" ] || [ -z "$VAST_PORT" ]; then
    usage
    exit 1
fi

case "$ARCHIVE_LEVEL" in
    best+final|all|best-only) ;;
    *)
        echo "Invalid archive level: $ARCHIVE_LEVEL" >&2
        usage
        exit 1
        ;;
esac

mkdir -p "$LOCAL_DIR/checkpoints" "$LOCAL_DIR/logs" "$LOCAL_DIR/results"

ssh_cmd() {
    ssh -p "$VAST_PORT" "root@${VAST_HOST}" "$@"
}

rsync_remote() {
    local remote_path="$1"
    local local_path="$2"
    mkdir -p "$(dirname "$local_path")"
    rsync "${RSYNC_FLAGS[@]}" -e "ssh -p $VAST_PORT" "root@${VAST_HOST}:${remote_path}" "$local_path"
}

highest_checkpoint_rel() {
    local variant="$1"
    ssh_cmd "cd /workspace/TALH && python3 - <<'PY'
from pathlib import Path
import re

variant = '${variant}'
ckpt_dir = Path('${REMOTE_ROOT}/checkpoints') / f'talh_{variant}'
paths = list(ckpt_dir.glob('step_*.pt')) + list(ckpt_dir.glob('step_*.safetensors'))
if not paths:
    raise SystemExit(0)

def step_num(path: Path) -> int:
    m = re.search(r'step_(\\d+)', path.name)
    return int(m.group(1)) if m else -1

best = max(paths, key=step_num)
print(best)
PY"
}

echo "======================================================"
echo "=== Sync Medium Budget Archive ==="
echo "=== From: ${VAST_HOST}:${VAST_PORT} ==="
echo "=== Policy: ${ARCHIVE_LEVEL} ==="
echo "======================================================"

echo ""
echo "[1/3] Syncing checkpoints..."
for variant in "${VARIANTS[@]}"; do
    remote_dir="${REMOTE_ROOT}/checkpoints/talh_${variant}"
    local_dir="${LOCAL_DIR}/checkpoints/talh_${variant}"
    mkdir -p "$local_dir"

    echo "  Variant: $variant"
    rsync_remote "${remote_dir}/config.json" "${local_dir}/config.json" || true
    rsync_remote "${remote_dir}/best.pt" "${local_dir}/best.pt" || true

    case "$ARCHIVE_LEVEL" in
        best-only)
            ;;
        best+final)
            highest="$(highest_checkpoint_rel "$variant" || true)"
            if [ -n "${highest:-}" ]; then
                file_name="$(basename "$highest")"
                rsync_remote "$highest" "${local_dir}/${file_name}" || true
            fi
            ;;
        all)
            rsync "${RSYNC_FLAGS[@]}" \
                --include='step_*.pt' \
                --include='step_*.safetensors' \
                --exclude='*' \
                -e "ssh -p $VAST_PORT" \
                "root@${VAST_HOST}:${remote_dir}/" \
                "${local_dir}/" || true
            ;;
    esac
done

echo ""
echo "[2/3] Syncing logs..."
rsync "${RSYNC_FLAGS[@]}" -e "ssh -p $VAST_PORT" \
    "root@${VAST_HOST}:${REMOTE_ROOT}/logs/" \
    "${LOCAL_DIR}/logs/" || true

echo ""
echo "[3/3] Syncing results..."
rsync "${RSYNC_FLAGS[@]}" -e "ssh -p $VAST_PORT" \
    "root@${VAST_HOST}:/workspace/TALH/experiments/medium/results/" \
    "${LOCAL_DIR}/results/" || true

echo ""
echo "======================================================"
echo "=== Sync Complete ==="
echo "======================================================"
echo "Local archive: ${LOCAL_DIR}"
echo "Variants archived: ${VARIANTS[*]}"
