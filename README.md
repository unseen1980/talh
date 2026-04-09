# TALH at 150M: Replication Package

Replication code and results for the paper:

**TALH at 150M: Ablating a Hybrid MLA+SSM Architecture for Edge Language Model Inference on Apple Silicon**  
Christos Koutsiaris, 2026

---

## What this repo contains

| Folder | Contents |
|---|---|
| `talh/` | Model source code (PyTorch): architecture, layers, training loop, checkpoint conversion |
| `configs/` | Exact JSON configs used for Phase 1 (all 5 variants) and Phase 2 (extended training) |
| `scripts/` | Training launch scripts, environment setup, local inference benchmark, evaluation |
| `results/phase1_logs/` | Training loss logs for all 5 variants (compute-matched, steps 0-4000) |
| `results/validation/` | Validation perplexity results at 512 and 1024 token context |
| `results/nano_pilot/` | 15M pilot experiment results (KV-cache compression, memory scaling) |
| `figures/` | Python script that regenerates all paper figures from the result data |

Trained checkpoints are not included (size). Running the training scripts as described below reproduces them.

---

## Architecture

TALH combines three components in each layer:

- **Multi-head Latent Attention (MLA)**: compresses the KV cache via a low-rank factorisation (`latent_dim=200` vs `d_model=800`, 4x theoretical compression)
- **SSM branch**: a parallel state space model with fixed-size hidden state, O(n) compute in sequence length
- **Gating**: a learned elementwise blend of MLA and SSM outputs

The five ablation variants explored in the paper:

| Variant | Components | Total params | Active params/token |
|---|---|---|---|
| `full` | MLA + SSM + ternary MoE | 493M | ~131M |
| `ssm_only` | SSM + MoE (no MLA) | 480M | ~131M |
| `mla_only` | MLA + MoE (no SSM) | 453M | ~131M |
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

### Step 1: Phase 1 training (compute-matched, all 5 variants)

```bash
bash scripts/train_phase1.sh
```

Trains all five variants for 4,000 steps with an identical budget:
- Steps 0-2999: 256-token sequences
- Steps 3000-4000: 512-token sequences
- Batch size 4, gradient accumulation 8 (effective batch 32)
- AdamW, cosine decay, LR 1e-4, 400 warmup steps, grad clip 0.5, BF16

This is the **primary comparison** in the paper. All architectural conclusions are drawn from these compute-matched checkpoints.

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

Reads the training logs and result CSVs from `../results/` and writes all paper figures to `figures/`. The TTFT latency numbers are hardcoded in `gen_figures.py` (lines 91-97) as they were measured manually on a MacBook M3.

### Step 4: Inference benchmark on Apple Silicon

After converting a checkpoint to MLX format:

```bash
python talh/convert_checkpoint.py --checkpoint experiments/medium_budget/checkpoints/talh_dense_ffn/ckpt_step4000.pt --output mlx_dense_ffn/

python scripts/benchmark_local_inference.py --model mlx_dense_ffn/ --context_lengths 512 1024 2048
```

This measures TTFT at the three context lengths reported in Table 7 of the paper. Run without quantisation for numbers comparable to the paper.

---

## Pre-computed results

The `results/` folder contains the raw data behind all paper tables:

- `results/phase1_logs/train_*.csv` — step-by-step training loss for all 5 variants (Tables 3, 4, 6)
- `results/validation/edge_0p5b_validation.csv` — validation perplexity at 512 and 1024 context (Table 5)
- `results/nano_pilot/kv_cache_vs_context.csv` — 15M pilot KV-cache compression measurements (Table 1, Fig. 1)
- `results/nano_pilot/talh_live_memory.csv` — 15M pilot peak memory vs context length (Table 1, Fig. 2)

TTFT measurements (Table 7) were collected with `scripts/benchmark_local_inference.py` on an M3 MacBook (18 GB unified memory) and are reported in `figures/gen_figures.py`.

---

## Key findings

1. **SSM and MLA serve orthogonal roles.** The SSM branch drives language modelling quality: `ssm_only` achieves PPL 239 vs 263 for the standard Transformer baseline, while `mla_only` is the worst-performing variant (PPL 315). MLA drives inference efficiency rather than quality.

2. **MLA's KV-cache compression produces near-linear TTFT scaling on Apple Silicon.** `mla_only` TTFT grows 1.32x from 512 to 2,048 tokens (4x context increase). All other variants grow 2.7-5.5x.

3. **Ternary MoE does not help at sub-200M active parameters.** `dense_ffn` (171M, all active) outperforms `full` (493M total, 131M active) at every training checkpoint while using 3.87 GB less GPU memory.

---

## Citation

```
@misc{koutsiaris2026talh,
  title        = {{TALH} at 150{M}: Ablating a Hybrid {MLA}+{SSM} Architecture
                  for Edge Language Model Inference on Apple Silicon},
  author       = {Koutsiaris, Christos},
  year         = {2026},
  note         = {Preprint}
}
```
