#!/bin/bash
# ---------------------------------------------------------------
# Prune old medium_budget checkpoints safely.
#
# Run on the Vast instance or locally inside the same workspace.
#
# Default behavior is dry-run.
# ---------------------------------------------------------------
set -euo pipefail

ROOT="/workspace/TALH"
if [ ! -d "$ROOT" ]; then
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$ROOT"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-experiments/medium_budget/checkpoints}"
LOG_ROOT="${LOG_ROOT:-experiments/medium_budget/logs}"
APPLY=0
PROTECT_RECENT_SECS="${PROTECT_RECENT_SECS:-600}"
VARIANTS=(full dense_ffn baseline mla_only ssm_only transformer_dense)

usage() {
    cat <<'EOF'
Usage:
  bash deploy/prune_medium_budget.sh [--apply] [--protect-recent-secs N]

Behavior:
  - keeps config.json
  - keeps best.pt / best.safetensors
  - keeps the highest step_* checkpoint per variant
  - prunes older step_* checkpoints unless they are too recent

Default mode is dry-run.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --apply)
            APPLY=1
            shift
            ;;
        --protect-recent-secs)
            PROTECT_RECENT_SECS="${2:-}"
            shift 2
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

active_log="$(ls -1t "${LOG_ROOT}"/*_stdout.log 2>/dev/null | head -n 1 || true)"
active_variant=""
active_phase=""
if [ -n "$active_log" ]; then
    base="$(basename "$active_log")"
    case "$base" in
        *_baseline_stdout.log)
            active_variant="${base%_baseline_stdout.log}"
            active_phase="baseline"
            ;;
        *_extend_stdout.log)
            active_variant="${base%_extend_stdout.log}"
            active_phase="extend"
            ;;
        *_retrieval_stdout.log)
            active_variant="${base%_retrieval_stdout.log}"
            active_phase="retrieval"
            ;;
        *_stdout.log)
            active_variant="${base%_stdout.log}"
            active_phase="budget"
            ;;
    esac
fi

python3 - <<'PY' "$CHECKPOINT_ROOT" "$PROTECT_RECENT_SECS" "$APPLY" "$active_variant" "$active_phase" "${VARIANTS[@]}"
import os
import re
import sys
import time
from pathlib import Path

checkpoint_root = Path(sys.argv[1])
protect_recent_secs = int(sys.argv[2])
apply = bool(int(sys.argv[3]))
active_variant = sys.argv[4]
active_phase = sys.argv[5]
variants = sys.argv[6:]

now = time.time()
total_bytes = 0
actions = []

def step_num(path: Path) -> int:
    m = re.search(r"step_(\d+)", path.name)
    return int(m.group(1)) if m else -1

for variant in variants:
    ckpt_dir = checkpoint_root / f"talh_{variant}"
    if not ckpt_dir.exists():
        continue

    step_files = sorted(
        list(ckpt_dir.glob("step_*.pt")) + list(ckpt_dir.glob("step_*.safetensors")),
        key=step_num,
    )
    keep = {ckpt_dir / "config.json", ckpt_dir / "best.pt", ckpt_dir / "best.safetensors"}
    if step_files:
        keep.add(step_files[-1])

    for path in step_files:
        if path in keep:
            continue

        age = now - path.stat().st_mtime
        if age < protect_recent_secs:
            actions.append((variant, "skip_recent", path, path.stat().st_size))
            continue

        if variant == active_variant and active_phase in {"budget", "baseline", "extend", "retrieval"} and path == step_files[-1]:
            actions.append((variant, "skip_anchor", path, path.stat().st_size))
            continue

        actions.append((variant, "delete", path, path.stat().st_size))
        total_bytes += path.stat().st_size

print("======================================================")
print("=== Medium Budget Prune Plan ===")
print("======================================================")
print(f"Mode: {'apply' if apply else 'dry-run'}")
print(f"Protect recent: {protect_recent_secs}s")
print(f"Active variant: {active_variant or 'unknown'}")
print(f"Active phase: {active_phase or 'unknown'}")
print("")

for variant, action, path, size in actions:
    print(f"{action:11s} | {variant:17s} | {size / (1024**3):6.2f} GB | {path}")

print("")
print(f"Potential reclaimed space: {total_bytes / (1024**3):.2f} GB")

if apply:
    deleted = 0
    for variant, action, path, _size in actions:
        if action != "delete":
            continue
        path.unlink(missing_ok=True)
        deleted += 1
    print(f"Deleted files: {deleted}")
print("======================================================")
PY
