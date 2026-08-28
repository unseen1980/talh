#!/bin/bash
# ---------------------------------------------------------------
# Vast.ai Instance Bootstrap — TALH FineWeb 10B Training
#
# Run this ONCE after SSH-ing into a new Vast.ai instance.
# It clones/updates the repo, installs dependencies, verifies
# the GPU, and runs a quick smoke test.
#
# Prerequisites:
#   - Set REPO_URL env var or edit the default below
#   - Optionally set WANDB_API_KEY for experiment tracking
#
# Usage:
#   export REPO_URL="https://github.com/<you>/TALH.git"
#   export WANDB_API_KEY="your_key"       # optional
#   bash deploy/setup.sh
#
# Or if repo already cloned to /workspace/TALH:
#   cd /workspace/TALH && bash deploy/setup.sh
# ---------------------------------------------------------------
set -euo pipefail

echo "=========================================="
echo "=== TALH Vast.ai Setup — $(date) ==="
echo "=========================================="

REPO_DIR="/workspace/TALH"
REPO_URL="${REPO_URL:-}"

# --- 1. Clone or update repo ---
echo ""
echo "[1/6] Setting up repository..."
if [ -d "$REPO_DIR/.git" ]; then
    echo "  Repo exists, pulling latest..."
    cd "$REPO_DIR"
    git pull --ff-only || echo "  WARNING: git pull failed (may have local changes)"
elif [ -n "$REPO_URL" ]; then
    echo "  Cloning from $REPO_URL..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
else
    # Check if we're already inside the repo (e.g., manually copied)
    if [ -f "/workspace/TALH/talh/model.py" ]; then
        echo "  Repo files found at $REPO_DIR (no .git dir)"
        cd "$REPO_DIR"
    else
        echo "ERROR: No repo found and REPO_URL not set."
        echo ""
        echo "  Option 1: Set REPO_URL and re-run:"
        echo "    export REPO_URL='https://github.com/<you>/TALH.git'"
        echo "    bash deploy/setup.sh"
        echo ""
        echo "  Option 2: Copy repo manually to $REPO_DIR"
        echo "    scp -r -P <port> ./TALH root@<host>:/workspace/TALH"
        echo "    bash deploy/setup.sh"
        exit 1
    fi
fi

cd "$REPO_DIR"

# --- 2. System-level dependencies ---
echo ""
echo "[2/6] Installing system dependencies..."

# Check for nvcc (needed for mamba-ssm compilation)
HAS_NVCC=false
if command -v nvcc &>/dev/null; then
    HAS_NVCC=true
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    echo "  nvcc found: $NVCC_VERSION"
else
    echo "  nvcc NOT found — checking for CUDA toolkit..."
    # Many vast.ai images have CUDA but nvcc is not on PATH
    for CUDA_DIR in /usr/local/cuda /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda-11.8; do
        if [ -f "$CUDA_DIR/bin/nvcc" ]; then
            export PATH="$CUDA_DIR/bin:$PATH"
            export LD_LIBRARY_PATH="${CUDA_DIR}/lib64:${LD_LIBRARY_PATH:-}"
            HAS_NVCC=true
            echo "  Found nvcc at $CUDA_DIR/bin/nvcc"
            # Persist for future shells
            echo "export PATH=$CUDA_DIR/bin:\$PATH" >> ~/.bashrc
            echo "export LD_LIBRARY_PATH=$CUDA_DIR/lib64:\${LD_LIBRARY_PATH:-}" >> ~/.bashrc
            break
        fi
    done
    if [ "$HAS_NVCC" = false ]; then
        echo "  WARNING: nvcc not found anywhere. mamba-ssm will NOT be available."
        echo "  Will use --ssm_backend minimal (pure PyTorch SSM)."
    fi
fi

# Ensure essential tools
apt-get update -qq && apt-get install -y -qq tmux htop rsync 2>/dev/null || true

# --- 3. Python dependencies ---
echo ""
echo "[3/6] Installing Python dependencies..."
pip install --quiet --upgrade pip setuptools 2>/dev/null || true
pip install --quiet --upgrade wheel 2>/dev/null || true

# Core deps
pip install --quiet -r requirements.txt

# Additional deps for training
pip install --quiet wandb safetensors

# Try to install mamba-ssm (needs nvcc + ninja)
SSM_BACKEND="minimal"
if [ "$HAS_NVCC" = true ]; then
    echo "  Attempting mamba-ssm install (requires compilation)..."
    pip install --quiet ninja 2>/dev/null || true
    if pip install --quiet "mamba-ssm>=2.2.0" 2>/dev/null; then
        echo "  mamba-ssm installed successfully!"
        SSM_BACKEND="mamba2"
    else
        echo "  WARNING: mamba-ssm installation failed."
        echo "  Using --ssm_backend minimal instead."
    fi
else
    echo "  Skipping mamba-ssm (no nvcc). Using minimal SSM backend."
fi

# Verify critical imports
echo ""
echo "  Verifying imports..."
python3 -c "
import torch, datasets, transformers, safetensors
print(f'  torch={torch.__version__}')
print(f'  datasets={datasets.__version__}')
print(f'  transformers={transformers.__version__}')
try:
    import mamba_ssm
    print(f'  mamba_ssm={mamba_ssm.__version__}')
except ImportError:
    print('  mamba_ssm=NOT INSTALLED (using minimal backend)')
try:
    import wandb
    print(f'  wandb={wandb.__version__}')
except ImportError:
    print('  wandb=NOT INSTALLED')
"

# --- 4. WandB login (optional) ---
echo ""
echo "[4/6] Configuring WandB..."
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
    echo "  WandB authenticated."
else
    echo "  WANDB_API_KEY not set — WandB logging disabled."
    echo "  To enable: export WANDB_API_KEY=<your_key>"
fi

# --- 5. GPU verification ---
echo ""
echo "[5/6] Verifying GPU..."
python3 -c "
import torch

if not torch.cuda.is_available():
    raise RuntimeError('CUDA not available — cannot train on this instance!')

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    vram_gb = (getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)) / 1e9
    print(f'  GPU {i}: {props.name}')
    print(f'  VRAM:   {vram_gb:.1f} GB')
    print(f'  CUDA:   {torch.version.cuda}')
    print(f'  BF16:   {torch.cuda.is_bf16_supported()}')

    if vram_gb < 20:
        print(f'  WARNING: {vram_gb:.1f} GB may be insufficient for medium model.')
    elif vram_gb >= 40:
        print(f'  EXCELLENT: A100-class GPU detected.')
    else:
        print(f'  OK: Sufficient for medium model (may need smaller batch).')
"

# --- 6. Smoke test ---
echo ""
echo "[6/6] Running smoke test (200 steps, TinyStories)..."
python3 -m talh.train_torch \
    --d_model 768 --n_layers 12 --num_heads 12 \
    --latent_dim 192 --state_dim 32 --d_ff 1536 \
    --vocab_size 50257 \
    --max_steps 200 --batch_size 4 --eval_every 100 --save_every 200 \
    --use_amp --gradient_checkpointing \
    --ssm_backend "$SSM_BACKEND" \
    --dataset tinystories \
    --output_dir ./experiments/medium/checkpoints/smoke_test \
    --warmup_steps 50

echo ""
echo "=========================================="
echo "=== Setup complete! ==="
echo "=========================================="
echo ""
echo "SSM Backend: $SSM_BACKEND"
echo ""
echo "Next steps:"
echo "  1. Start a tmux session:    tmux new -s train"
echo "  2. Start training:          bash deploy/train_medium.sh"
echo "  3. Detach tmux:             Ctrl-B D"
echo "  4. Monitor from outside:    bash deploy/monitor.sh"
echo "  5. Check WandB dashboard:   https://wandb.ai"