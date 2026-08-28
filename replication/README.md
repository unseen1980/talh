# TALH Exploratory Ablation: Replication Package

Replication code and results for the paper:

**An Exploratory Ablation of a Small MLA--SSM Hybrid Language Model**
Christos Koutsiaris, 2026

---

## What this repo contains

| Folder | Contents |
|---|---|
| `talh/` | Model source code (PyTorch): architecture, layers, training loop, checkpoint conversion |
| `configs/` | Exact JSON configs used for Phase 1 (all 5 variants) and Phase 2 (extended training) |
| `scripts/` | Training launch scripts, environment setup, local inference benchmark, evaluation |
| `results/phase1_logs/` | Training loss logs for all 5 variants (same steps and nominal tokens) |
| `results/validation/` | Validation perplexity results at 512 and 1024 token context |
| `results/nano_pilot/` | 15M pilot experiment results (KV-cache compression, memory scaling) |
| `figures/` | Python script that regenerates all paper figures from the result data |

Trained checkpoints are not included (size). Running the training scripts as described below reproduces them.

---

## Architecture

TALH combines three components in each layer:

- **Multi-head Latent Attention (MLA)**: compresses the KV cache via a low-rank factorisation (`latent_dim=200` vs `d_model=800`, 4x theoretical compression)
- **Recurrent branch**: a custom minimal selective recurrence with fixed-size hidden state; it is not Mamba/Mamba-2
- **Gating**: a learned elementwise blend of MLA and SSM outputs

The five ablation variants explored in the paper:

| Variant | Components | Total params | Active params/token |
|---|---|---|---|
| `full` | MLA + SSM + ternary MoE | 493M | ~217M |
| `ssm_only` | SSM + MoE (no MLA) | 480M | ~203M |
| `mla_only` | MLA + MoE (no SSM) | 453M | ~177M |
| `dense_ffn` | MLA + SSM + dense FFN | 171M | 171M |
| `transformer_dense` | MHA + dense FFN (baseline) | 117M | 117M |

---

## Requirements

**Training (GPU, tested on A100-40GB via Vast.ai):**

```bash
pip install -r requirements.txt
# PyTorch with CUDA is required for training
# Install CUDA-specific packages manually on the GPU instance:
# pip install mamba-ssm  # optional, falls back to minimal SSM
```

**Inference benchmark (Apple Silicon Mac):**

```bash
pip install mlx mlx-lm transformers
```

---

## Replication steps

### Step 0: Environment setup

On a GPU instance (Vast.ai or equivalent, A100-40GB recommended):

```bash
bash scripts/setup_env.sh
```

This installs dependencies and configures the FineWeb dataset path. The training scripts use the `fineweb` dataset from Hugging Face (`HuggingFaceFW/fineweb`, `sample-10BT` subset) and cache parquet shards locally.

### Step 1: Phase 1 training (step- and token-matched, all 5 variants)

```bash
bash scripts/train_phase1.sh
```

Trains all five variants for 4,000 steps with an identical budget:
- Steps 0-2999: 256-token sequences
- Steps 3000-4000: 512-token sequences
- Batch size 4, gradient accumulation 8 (effective batch 32)
- AdamW, cosine decay, LR 1e-4, 400 warmup steps, grad clip 0.5, BF16

This is the primary comparison in the paper. It matches steps and nominal
tokens, not parameter counts, FLOPs, or wall-clock time.

Expected runtime: ~8-10 hours per variant on a single A100-40GB.

### Step 2: Phase 2 extended training (full + dense_ffn only)

```bash
bash scripts/train_phase2.sh
```

Extends `full` and `dense_ffn` to 7,000 steps, adding a 1,024-token curriculum stage (steps 5000-7000). Batch size is reduced from 4 to 2 (gradient accumulation doubled to 16) for the 1024-token stage to avoid OOM.

**Note:** Phase 2 results cannot be directly compared against the three variants that were not extended. See the Limitations section of the paper.

### Step 3: Reproduce figures

```bash
cd figures
python gen_figures.py
```

The script writes the paper figures. Several arrays, including the historical
TTFT observations, are embedded in the script. The TTFT data were recorded
manually on a MacBook M3; raw repeated-trial records are unavailable.

### Step 4: Inference benchmark on Apple Silicon

The benchmark CLI accepts a checkpoint root and variant list:

```bash
python scripts/benchmark_local_inference.py \
  --checkpoints_root experiments/medium_budget/checkpoints \
  --variants full dense_ffn mla_only ssm_only transformer_dense \
  --prompt_lens 512 1024 2048 \
  --backend mlx
```

Paper-scale checkpoints are not distributed, and the current MLX loader needs
architecture-aware handling for all ablation variants. Consequently this
command documents the intended harness but does not, by itself, reproduce the
historical table. A future release should archive checkpoints, environment
versions, warm-up policy, repeated trials, and raw latency/memory CSV files.

---

## Pre-computed results

The `results/` folder contains the raw data behind all paper tables:

- `results/phase1_logs/train_*.csv` — step-by-step training loss for all 5 variants (Tables 3, 4, 6)
- `results/validation/edge_0p5b_validation.csv` — validation perplexity at 512 and 1024 context (Table 5)
- `results/nano_pilot/kv_cache_vs_context.csv` — 15M pilot KV-cache compression measurements (Table 1, Fig. 1)
- `results/nano_pilot/talh_live_memory.csv` — 15M pilot peak memory vs context length (Table 1, Fig. 2)

TTFT observations (Table 7) are reported in `figures/gen_figures.py`; the raw
measurement records and paper-scale checkpoints are not included.

---

## Key findings

1. **Component-ablation hypothesis.** In this run, removing the custom recurrent branch hurts validation loss most: `ssm_only` obtains PPL 239 while `mla_only` obtains PPL 315. This is implementation-specific and single-seed.

2. **Preliminary TTFT observation.** `mla_only` has the flattest historical TTFT curve, growing 1.32x from 512 to 2,048 tokens. Without repetitions or memory measurements, this does not establish a scaling law or isolate KV-cache compression as the cause.

3. **Negative result for one ternary-MoE setup.** `dense_ffn` obtains lower loss than `full` (~217M active, 493M total) at every reported checkpoint while using 3.87 GB less peak training memory. The experiment does not establish a general MoE scale threshold.

---

## Citation

```
@misc{koutsiaris2026talh,
  title        = {An Exploratory Ablation of a Small {MLA}--{SSM}
                  Hybrid Language Model},
  author       = {Koutsiaris, Christos},
  year         = {2026},
  note         = {Preprint}
}
```
