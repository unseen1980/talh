# TALH Project Briefing

> Current source of truth: `paper/talh_150m_ablation.tex` and
> `paper/talh_150m_ablation.pdf`.
>
> This briefing is aligned with the current paper. Older notes described an
> unfinished 456M/768d medium run and a remote continuation workflow. That
> state is stale.

> **2026 review correction.** The paper has been retitled **An Exploratory
> Ablation of a Small MLA--SSM Hybrid Language Model**. The historical filename
> is retained for stable links. Correct estimated active counts are 217M
> (`full`), 203M (`ssm_only`), 177M (`mla_only`), 171M (`dense_ffn`), and 117M
> (`transformer_dense`). Phase 1 matches steps and nominal tokens, not compute.
> Historical TTFT values lack raw repeated-trial records and are preliminary.
> Where this older briefing conflicts with the paper or `replication/README.md`,
> those two files take precedence.

---

## 1. What TALH Is

TALH is an experimental decoder-only language-model architecture for local and
edge inference. The current paper expands TALH as **Adaptive Latent Hybrid**.
Earlier notes used "Ternary Adaptive Latent Hybrid", but the paper now treats
ternary MoE as one ablation component, not as the recommended final design.

The current paper is:

- `paper/talh_150m_ablation.tex`
- `paper/talh_150m_ablation.pdf`

The revised paper title is:

**An Exploratory Ablation of a Small MLA--SSM Hybrid Language Model**

The project is best understood as an exploratory single-seed ablation, not a claim of a
state-of-the-art model. The main question is how Multi-head Latent Attention
(MLA), a custom minimal recurrent branch, and feed-forward/MoE choices behave
across variants with approximately 117--217M active parameters per token.

---

## 2. Current Paper Claim

The paper records two implementation-specific hypotheses and one negative result.

1. **Removing the recurrent branch hurts validation loss most in this run.**
   This motivates a component-role hypothesis but does not establish a general
   decomposition across SSM and MLA architectures.

2. **`mla_only` has the flattest preliminary TTFT curve.**
   Its historical TTFT grows 1.32x when prompt length grows from 512 to 2,048
   tokens, but absent repetitions and memory measurements this does not prove a
   scaling law or isolate KV-cache compression as the cause.

3. **Ternary MoE is a negative result at this scale.**
   The dense FFN hybrid beats the full ternary MoE variant while using less
   training memory. At this active-parameter scale, the experts do not appear to
   specialize enough to justify the stored-weight and memory overhead.

---

## 3. Architecture in the Current Paper

Each TALH hybrid layer combines:

- **MLA branch**: compresses the KV cache by storing a low-dimensional latent
  vector instead of full key/value tensors.
- **SSM branch**: processes sequence information through a recurrent state
  update with linear-time sequence behavior.
- **Gated fusion**: learns how to combine the MLA and SSM outputs.
- **Feed-forward block**: either ternary sparse MoE or dense SwiGLU depending
  on the ablation variant.

The `full` variant includes ternary MoE. The best-performing hybrid in the
paper is `dense_ffn`, which replaces ternary MoE with a standard dense BF16
SwiGLU FFN.

---

## 4. Current Main Configuration

The paper-facing 0.5B-class setup is defined by:

- `configs/medium_0p5b_budget.json`
- `configs/medium_0p5b_edge_followup.json`
- mirrored copies under `replication/configs/`

Key model settings:

| Setting | Value |
| --- | --- |
| Vocabulary | GPT-2 tokenizer, 50,257 tokens |
| Layers | 12 |
| Hidden size | 800 |
| Attention heads | 16 |
| MLA latent dimension | 200 |
| SSM state dimension | 32 |
| Experts | 8 |
| MoE top-k | 2 |
| FFN width | 1,600 |

Important: `configs/medium.json` is a legacy plan for the older 768d/456M
experiment. It is not the paper-facing configuration.

---

## 5. Ablation Variants

The current paper compares five variants:

| Variant | Components | Total params | Active params/token |
| --- | --- | ---: | ---: |
| `full` | MLA + SSM + ternary MoE | 493M | ~217M |
| `ssm_only` | SSM + MoE, no MLA | 480M | ~203M |
| `mla_only` | MLA + MoE, no SSM | 453M | ~177M |
| `dense_ffn` | MLA + SSM + dense FFN | 171M | 171M |
| `transformer_dense` | Standard MHA + dense FFN | 117M | 117M |

The historical "150M" framing was based on incomplete active-parameter
accounting and has been removed from the revised title. See the corrected
counts in the table above.

---

## 6. Experimental Setup

### 6.1 Preliminary 15M Experiment

The 15M pilot was used to confirm local runnability and measure cache/memory
behavior before the larger ablation study.

Evidence files:

- `experiments/nano/results/kv_cache_vs_context.csv`
- `experiments/nano/results/talh_live_memory.csv`
- `experiments/nano/results/talh_results.png`
- copied paper figures under `paper/figures/`

Key findings:

- MLA measured an 8x KV-cache reduction relative to standard MHA.
- Peak memory stayed within the 18GB Apple Silicon budget up to 4,096-token
  context.
- The 15M pilot reached TinyStories perplexity of about 1.93 for `full` and
  1.49 for `dense_ffn`, against a GPT-Neo reference around 3.00.

The 15M model was too small for reliable retrieval behavior. Earlier NIAH and
ARC probes returned 0 percent accuracy and should be interpreted as scale
limitations, not as downstream success.

### 6.2 Main Training

Dataset:

- FineWeb `sample-10BT`
- GPT-2 tokenizer
- Held-out validation pool described in the paper

Hardware:

- Single NVIDIA A100-SXM4-40GB GPU

Training settings:

| Setting | Phase 1 budget scan | Phase 2 extension |
| --- | --- | --- |
| Variants | all five variants | `full` and `dense_ffn` |
| Max steps | 4,000 | 7,000 |
| Curriculum | 256 -> 512 tokens | 256 -> 512 -> 1,024 tokens |
| Batch / accumulation | 4 / 8 | 2 / 16 for 1,024-token stage |
| Optimizer | AdamW | AdamW |
| LR | 1e-4 | 1e-4 |
| Warmup | 400 steps | 400 steps |
| Gradient clip | 0.5 | 0.5 |
| Precision | BF16 mixed precision | BF16 mixed precision |
| SSM backend | minimal | minimal |
| Gradient checkpointing | off | off |

Phase 1 matches optimisation steps and nominal tokens, not FLOPs, parameter
counts, or wall-clock time. It is the primary comparison point. Phase 2 extends
only `full` and `dense_ffn`, so those numbers must not be compared as if all five
variants received the same budget.

---

## 7. Current Results

### 7.1 Phase 1: Step- and Token-Matched Quality at 4,000 Steps

From the paper's Phase 1 table:

| Variant | Val loss | PPL | GPU memory |
| --- | ---: | ---: | ---: |
| `dense_ffn` | 5.443 | 231 | 24.01 GB |
| `ssm_only` | 5.476 | 239 | 26.50 GB |
| `full` | 5.480 | 240 | 27.88 GB |
| `transformer_dense` | 5.572 | 263 | 5.36 GB |
| `mla_only` | 5.753 | 315 | 16.20 GB |

Interpretation:

- `dense_ffn` has the lowest measured loss, but the variants are not matched in
  active parameters or training cost.
- `ssm_only` remains close to the full ternary-MoE hybrid, motivating the
  hypothesis that the recurrent branch matters more for loss in this setup.
- `mla_only` has the weakest perplexity in this single run.
- `transformer_dense` is much smaller and faster, but has weaker quality than
  the best hybrid under this configuration.

### 7.2 Phase 2: Extended Training for `full` and `dense_ffn`

At 7,000 steps:

| Variant | Val loss at 7K | Context |
| --- | ---: | --- |
| `dense_ffn` | 5.074 | 1,024 |
| `full` | 5.093 | 1,024 |

`dense_ffn` remains slightly ahead of `full`, and it uses about 3.87GB less
training memory at the 1,024-token stage.

### 7.3 Validation CSV

The current validation summary is:

- `replication/results/validation/edge_0p5b_validation.csv`

It reports best-checkpoint validation at 512 and 1,024 context lengths. Key
rows:

| Variant | PPL 512 | PPL 1024 |
| --- | ---: | ---: |
| `dense_ffn` | 154.33 | 160.32 |
| `full` | 156.96 | 163.73 |
| `ssm_only` | 241.31 | 250.97 |
| `transformer_dense` | 269.82 | 343.92 |
| `mla_only` | 318.82 | 359.53 |

These numbers are useful for the final result package, but the paper's primary
comparison is the step- and token-matched Phase 1 table.

### 7.4 Local Apple Silicon TTFT

From the paper:

| Variant | 512 tok | 1024 tok | 2048 tok |
| --- | ---: | ---: | ---: |
| `transformer_dense` | 114 ms | 256 ms | 631 ms |
| `mla_only` | 1,599 ms | 1,739 ms | 2,110 ms |
| `dense_ffn` | 2,560 ms | 4,508 ms | 8,390 ms |
| `ssm_only` | 3,312 ms | 5,100 ms | 8,643 ms |
| `full` | 3,364 ms | 5,221 ms | 9,050 ms |

Important interpretation:

- `transformer_dense` is fastest in absolute TTFT because standard MHA benefits
  from optimized MLX kernels and it has fewer operations.
- `mla_only` has the best scaling with context length. It grows only 1.32x
  from 512 to 2,048 tokens.
- Hybrid absolute latency is currently an implementation limitation. The paper
  is transparent that the SSM scan and MLA projections are not kernel-optimized.

---

## 8. Main Conclusions

The paper's current conclusions are:

1. **SSM drives quality.**
   `ssm_only` and the full/dense hybrid variants outperform the standard
   transformer baseline on perplexity, while `mla_only` performs worst.

2. **MLA drives latency scaling, not quality.**
   MLA's value is in cache compression and context-scaling behavior.

3. **Dense FFN is better than ternary MoE at this scale.**
   Ternary MoE adds memory overhead without quality benefit below 1B active
   parameters in this experiment.

4. **The current implementation is not optimized for absolute latency.**
   Standard MHA is much faster in absolute terms on MLX today because it uses
   optimized kernels. The hybrid needs SSM/MLA kernel work and quantization to
   become competitive for real edge deployment.

---

## 9. Limitations to Keep Visible

Do not overstate the result. The paper explicitly includes these limitations:

- Single training run per variant; no multi-seed replication.
- Parameter counts are not perfectly balanced.
- FFN width is 2x hidden size, not the common 4x transformer setting.
- Phase 1 is not FLOP-, parameter-, or wall-clock-matched; Phase 2 additionally
  extends only two variants.
- Results are perplexity and TTFT focused; no downstream task benchmark is
  reported.
- Inference benchmarking does not report decode throughput, peak inference
  memory, power, or production-optimized kernels.

---

## 10. Current Repository Map

Useful reader-facing files:

```text
TALH/
├── README.md
├── PROJECT_BRIEFING.md
├── paper/
│   ├── talh_150m_ablation.tex
│   ├── talh_150m_ablation.pdf
│   └── figures/
├── replication/
│   ├── README.md
│   ├── configs/
│   ├── results/
│   │   ├── nano_pilot/
│   │   ├── phase1_logs/
│   │   └── validation/
│   ├── scripts/
│   └── talh/
├── talh/
│   ├── model.py
│   ├── train_torch.py
│   ├── train_mlx.py
│   ├── convert_checkpoint.py
│   └── layers/
├── scripts/
├── configs/
├── deploy/
├── tests/
└── requirements.txt
```

Important note:

- `PROJECT_BRIEFING.md` is now a summary for humans.
- `paper/talh_150m_ablation.tex` is the authoritative paper text.
- `replication/README.md` is the best entry point for reproduction.
- Old remote server/SSH instructions should not be used as project truth.

---

## 11. What KPMG or an External Reader Should Read

Recommended reading order:

1. `README.md`
2. `PROJECT_BRIEFING.md`
3. `paper/talh_150m_ablation.pdf`
4. `replication/README.md`
5. `replication/results/validation/edge_0p5b_validation.csv`
6. `replication/results/phase1_logs/*.csv`
7. `paper/figures/`

This gives the project narrative, the experiment setup, the core findings, and
the supporting result artifacts without exposing local checkpoints or private
run operations.

---

## 12. Remaining Cleanup Before Public GitHub Push

The root `TALH/` folder currently has a nested Git repository under
`replication/.git`. If the root project is pushed as one GitHub repo, remove
the nested `.git` directory first so that `replication/` is published as normal
files:

```bash
rm -rf replication/.git
```

Do not commit:

- model checkpoints
- local datasets
- `.env` or credentials
- private remote host details
- Python caches
- LaTeX build artifacts

The root `.gitignore` is set up for this cleaned public push.

---

## 13. Short Version

TALH is now a paper-facing exploratory ablation, not an in-progress server
training run. In the archived single-seed experiment, the dense-FFN hybrid has
the lowest measured loss, removing the custom recurrent branch hurts most, and
the tested ternary MoE gives no quality benefit. The historical MLA-only TTFT
curve is preliminary because repeated raw measurements are unavailable.
