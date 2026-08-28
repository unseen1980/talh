# TALH Medium Training — Vast.ai Runbook

## Overview

Train 5 ablation variants of the TALH architecture (~460M params, ~200M active) on **FineWeb 10B** using a Vast.ai A100 GPU instance. After training, download checkpoints and run inference locally on Mac (MLX) or CPU (PyTorch).

### Ablation Variants

| Variant | Description |
|---------|-------------|
| `full` | All components: MLA + SSM + Ternary MoE + Gated Fusion |
| `mla_only` | MLA attention only (SSM replaced with identity) |
| `ssm_only` | SSM only (MLA replaced with identity) |
| `dense_ffn` | Dense FFN instead of Ternary MoE |
| `baseline` | Standard dense transformer (no hybrid, no ternary) |

### Estimated Time & Cost

- ~10-14 hours per variant on A100 40GB
- ~40-52 hours total for all 5 variants
- ~$27-35 at $0.663/hr

---

## Quick Start (TL;DR)

```bash
# 1. SSH into vast.ai instance
ssh -p <PORT> root@<HOST>

# 2. Setup (once)
cd /workspace/TALH
bash deploy/setup.sh

# 3. Train (in tmux)
tmux new -s train
bash deploy/train_medium.sh
# Ctrl-B D to detach

# 4. Monitor (from another SSH session)
bash deploy/monitor.sh
watch -n 30 bash deploy/monitor.sh   # auto-refresh

# 5. Evaluate (after training completes)
bash deploy/evaluate_on_gpu.sh

# 6. Download to Mac (run FROM your Mac)
bash deploy/download_and_convert.sh <VAST_IP> <VAST_PORT>

# 7. DESTROY the instance to stop billing!
```

---

## Detailed Workflow

### Step 1: Provision Vast.ai Instance

- **GPU**: 1× A100 PCIE 40GB (or A100 SXM4 40GB)
- **Image**: PyTorch 2.x + CUDA 12.x
- **Disk**: ≥80 GB
- **Cost target**: ~$0.60-0.70/hr

### Step 2: Upload Code

```bash
# Option A: Git clone (set REPO_URL in setup.sh or export it)
export REPO_URL="https://github.com/<you>/TALH.git"

# Option B: SCP upload
scp -r -P <PORT> . root@<HOST>:/workspace/TALH/
```

### Step 3: Run Setup (`deploy/setup.sh`)

```bash
ssh -p <PORT> root@<HOST>
cd /workspace/TALH
bash deploy/setup.sh
```

**What it does:**
1. Clones/updates repo (or detects manually copied files)
2. Finds nvcc and adds to PATH (searches common CUDA paths)
3. Installs system deps (tmux, htop, rsync)
4. Installs Python deps from `requirements.txt` + wandb + safetensors
5. Attempts `mamba-ssm` install (graceful fallback to minimal SSM)
6. Verifies GPU (VRAM, CUDA version, BF16 support)
7. Runs 200-step smoke test on TinyStories

**If smoke test fails:**
- Missing package → `pip install <package>`
- CUDA OOM → reduce batch_size in smoke test command
- Import error → check `pip list` for missing deps

### Step 4: Train (`deploy/train_medium.sh`)

```bash
tmux new -s train
bash deploy/train_medium.sh          # all 5 variants
bash deploy/train_medium.sh full     # single variant only
```

Detach tmux: `Ctrl-B D`

**What it does:**
- Auto-detects SSM backend (mamba2 if available, else minimal)
- Auto-tunes batch_size/grad_accumulation based on GPU VRAM:
  - A100 40GB → batch=16, accum=2 (effective=32)
  - 24GB GPU → batch=8, accum=4 (effective=32)
  - <24GB → batch=4, accum=8 (effective=32)
- Trains each variant for 100K steps on FineWeb-10BT (streamed)
- Auto-skips completed variants (detects `step_100000.pt`)
- Auto-resumes from last checkpoint if interrupted
- Logs stdout to `experiments/medium/logs/<variant>_stdout.log`
- CSV metrics to `experiments/medium/logs/train_<variant>.csv`

**Key training settings:**
- Model: d_model=768, n_layers=12, num_heads=12, n_experts=8, top_k=2
- Optimizer: AdamW, lr=3e-4, warmup=2000 steps, cosine decay
- AMP (BF16) + gradient checkpointing enabled
- Checkpoints every 5K steps, keeps last 3 + best.pt
- Curriculum context extension: 512→1024→2048→4096

### Step 5: Monitor (`deploy/monitor.sh`)

```bash
bash deploy/monitor.sh              # one-shot status
watch -n 30 bash deploy/monitor.sh  # auto-refresh every 30s
```

**Shows:**
- GPU utilization, VRAM usage, temperature
- Active training process PID and uptime
- Per-variant progress: step, loss, val_loss, seq_len, ETA
- Best validation loss per variant
- Disk usage

### Current 0.5B `medium_budget` pipeline

The live paper-facing 0.5B pipeline uses `experiments/medium_budget/`, not `experiments/medium/`.

Use these commands instead:

```bash
# Run on the Vast instance
bash deploy/monitor_edge_pipeline.sh
watch -n 30 bash deploy/monitor_edge_pipeline.sh

# Run from your Mac
bash deploy/sync_medium_budget.sh <VAST_IP> <VAST_PORT>
```

Notes:
- `deploy/monitor_budget.sh` only reports the original 5-way budget scan and can be misleading once follow-up phases begin.
- `deploy/download_and_convert.sh` and `deploy/sync_checkpoints.sh` are legacy helpers for the older `experiments/medium/` workflow.
- `deploy/prune_medium_budget.sh` is the safe cleanup tool for reclaiming space after local verification.

### Step 6: GPU Evaluation (`deploy/evaluate_on_gpu.sh`)

Run **after training completes**, before destroying the instance:

```bash
bash deploy/evaluate_on_gpu.sh
```

**Produces:**
- `experiments/medium/results/validation_perplexity.csv` — perplexity at seq_len 512/1024/2048
- `experiments/medium/results/generation_samples.txt` — sample text generations
- `experiments/medium/results/training_summary.json` — final loss per variant

### Step 7: Download & Convert (`deploy/download_and_convert.sh`)

This section applies to the older `experiments/medium/` workflow.
For the current 0.5B paper pipeline, sync first with:

```bash
bash deploy/sync_medium_budget.sh <VAST_IP> <VAST_PORT>
```

Run **FROM YOUR MAC** (not the instance):

```bash
bash deploy/download_and_convert.sh <VAST_IP> <VAST_PORT>
```

**What it does:**
1. Rsyncs `best.pt` + `config.json` for all 5 variants (skips intermediate checkpoints)
2. Downloads training logs and evaluation results
3. Converts all `.pt` → `.safetensors` (MLX format) using `talh/convert_checkpoint.py`
4. Verifies loading on Apple Silicon (MLX) or CPU (PyTorch)

### Step 8: Destroy Instance

**IMPORTANT**: Destroy the Vast.ai instance immediately after download to stop billing!

### Step 9: Local Benchmarks (Mac)

```bash
# Memory vs context length
python scripts/benchmark_memory.py \
    --n_layers 12 --d_model 768 --num_heads 12 --latent_dim 192 \
    --seq_lens 256 512 1024 2048 4096 8192 \
    --output_dir experiments/medium/results/

# Latency benchmarks
python scripts/benchmark_latency.py \
    --model talh --n_layers 12 --d_model 768 --num_heads 12 --latent_dim 192 \
    --prompt_lens 256 512 1024 2048 4096 \
    --output_dir experiments/medium/results/
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | `pip install <module>` |
| CUDA OOM during training | Reduce batch_size: edit `train_medium.sh` or use smaller GPU tier settings |
| `mamba-ssm` fails to install | OK — script falls back to `--ssm_backend minimal` |
| Training hangs (GPU 0% util) | Kill process, check `nvidia-smi`, restart |
| SSH disconnects | Training runs in tmux — reconnect with `tmux attach -t train` |
| Checkpoint resume fails | Delete corrupt checkpoint, restart variant |
| `vocab_size` mismatch | Ensure `--vocab_size 50257` (GPT-2 tokenizer) |
| Disk full | For the current paper pipeline, use `bash deploy/prune_medium_budget.sh` first; do not blindly delete the active resume anchor. |

---

## File Structure

```
/workspace/TALH/
├── deploy/
│   ├── setup.sh                 # Instance bootstrap (run once)
│   ├── train_medium.sh          # Master training script
│   ├── monitor.sh               # Legacy monitor for experiments/medium
│   ├── monitor_edge_pipeline.sh # Follow-up monitor for experiments/medium_budget
│   ├── evaluate_on_gpu.sh       # GPU-side evaluation
│   ├── download_and_convert.sh  # Download to Mac + PT→MLX conversion
│   ├── sync_checkpoints.sh      # Legacy rsync script
│   ├── sync_medium_budget.sh    # Current paper-pipeline rsync helper
│   └── prune_medium_budget.sh   # Safe checkpoint cleanup for medium_budget
│   └── RUNBOOK.md               # This file
├── configs/
│   └── medium.json              # Model config reference
├── talh/
│   ├── train_torch.py           # PyTorch training script
│   ├── model.py                 # MLX model definition
│   ├── convert_checkpoint.py    # PT ↔ safetensors conversion
│   └── layers/                  # MLA, SSM, Ternary MoE, etc.
├── experiments/medium/
│   ├── checkpoints/             # Trained model weights
│   │   ├── talh_full/
│   │   ├── talh_mla_only/
│   │   ├── talh_ssm_only/
│   │   ├── talh_dense_ffn/
│   │   └── talh_baseline/
│   ├── logs/                    # CSV training logs + stdout logs
│   └── results/                 # Evaluation outputs
└── scripts/                     # Local benchmark scripts
