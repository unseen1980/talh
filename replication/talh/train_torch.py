"""
TALH Training Script — PyTorch / GPU (Vast.ai, Colab, or local).

Trains a TALH model on TinyStories or FineWeb using a curriculum that
starts with short contexts and gradually extends them.

This script intentionally mirrors the MLX model architecture so that
trained checkpoints can be ported back to MLX for Mac benchmarking.

Supports:
    - Mixed precision (BF16 AMP)
    - Gradient accumulation
    - Gradient checkpointing (activation checkpointing)
    - Mamba-2 CUDA SSM kernels
    - DistributedDataParallel (multi-GPU)
    - WandB experiment tracking
    - CSV training log (crash-resilient)

Usage:
    # Nano model (backward-compatible with original defaults)
    python -m talh.train_torch \\
        --dataset tinystories --n_layers 4 --d_model 256 \\
        --max_steps 50000 --output_dir ./experiments/nano/checkpoints/talh

    # Medium model on Vast.ai A100
    python -m talh.train_torch \\
        --dataset fineweb --d_model 768 --n_layers 12 --num_heads 12 \\
        --latent_dim 192 --state_dim 32 --d_ff 1536 \\
        --max_steps 100000 --batch_size 8 --grad_accumulation_steps 4 \\
        --use_amp --gradient_checkpointing --ssm_backend mamba2 \\
        --wandb_project talh-medium \\
        --output_dir ./experiments/medium/checkpoints/talh

Ablation variants (add --ablation <name>):
    full              — default, all components active
    no_ttt            — TTT disabled  (not applicable in pretraining; reserved)
    mla_only          — SSM branch replaced with identity pass-through
    ssm_only          — MLA branch replaced with identity pass-through
    dense_ffn         — Replace ternary MoE with standard dense SwiGLU FFN
    baseline          — Legacy dense MLA-only baseline (kept for compatibility)
    transformer_dense — Standard dense Transformer (MHA + dense FFN)
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,   # Colab shows stdout; stderr is often hidden
    force=True,          # override any previous basicConfig calls
)
logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    """Log + force-flush stdout so Colab shows progress immediately."""
    logger.info(msg)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model
    vocab_size:   int   = 32000
    n_layers:     int   = 4
    d_model:      int   = 256
    num_heads:    int   = 4
    latent_dim:   int   = 64
    state_dim:    int   = 16
    n_experts:    int   = 8
    top_k:        int   = 2
    d_ff:         int   = 512
    fusion:       str   = "gated"
    ablation:     str   = "full"   # full|no_ttt|mla_only|ssm_only|dense_ffn|baseline|transformer_dense

    # Training
    dataset:      str   = "tinystories"
    max_seq_len:  int   = 512
    batch_size:   int   = 32
    max_steps:    int   = 50_000
    eval_every:   int   = 500
    save_every:   int   = 5_000
    log_every:    int   = 100
    val_batches:  int   = 50
    lr:           float = 3e-4
    weight_decay: float = 0.1
    grad_clip:    float = 1.0
    warmup_steps: int   = 1_000
    aux_loss_coef:float = 0.01

    # Context curriculum (gradually increase sequence length)
    curriculum_steps: list[tuple[int, int]] = None  # [(step, seq_len), ...]

    # Scaling infrastructure
    grad_accumulation_steps: int  = 1       # micro-batches per optimizer step
    use_amp:                 bool = False   # BF16 mixed precision
    gradient_checkpointing:  bool = False   # activation checkpointing per layer
    use_ddp:                 bool = False   # DistributedDataParallel
    ssm_backend:             str  = "minimal"  # "minimal" or "mamba2"
    num_workers:             int  = -1
    val_num_workers:         int  = -1

    # Logging & tracking
    wandb_project: str = ""   # WandB project name (empty = disabled)
    log_csv:       str = ""   # CSV log path (empty = disabled)

    # I/O
    output_dir:   str   = "./checkpoints/talh"
    seed:         int   = 42

    # Checkpoint management
    keep_last_n:  int   = 3    # keep last N step checkpoints + best.pt

    def __post_init__(self):
        if self.curriculum_steps is None:
            self.curriculum_steps = [
                (0,       256),
                (5_000,   512),
                (12_500, 1024),
                (20_000, 2048),
            ]
        else:
            self.curriculum_steps = [
                (int(step), int(seq_len))
                for step, seq_len in self.curriculum_steps
            ]
        self.curriculum_steps = sorted(self.curriculum_steps, key=lambda item: item[0])
        if not self.curriculum_steps or self.curriculum_steps[0][0] != 0:
            raise ValueError("curriculum_steps must start at step 0")
        if self.num_workers < 0:
            self.num_workers = 0 if self.dataset == "fineweb" else 4
        if self.val_num_workers < 0:
            self.val_num_workers = 0 if self.dataset == "fineweb" else 2


def _parse_curriculum_spec(spec: str) -> list[tuple[int, int]]:
    """Parse a curriculum like '0:256,3000:512'."""
    steps: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        step_str, seq_str = chunk.split(":", maxsplit=1)
        steps.append((int(step_str), int(seq_str)))
    if not steps:
        raise ValueError("curriculum spec is empty")
    steps = sorted(steps, key=lambda item: item[0])
    if steps[0][0] != 0:
        raise ValueError("curriculum must start at step 0")
    return steps


# ---------------------------------------------------------------------------
# PyTorch model building blocks
# ---------------------------------------------------------------------------

def ternary_quantize(w: torch.Tensor) -> torch.Tensor:
    """Threshold-based ternary quantization to {-1, 0, +1}."""
    thresh = 0.5 * w.abs().mean()
    return (w > thresh).float() - (w < -thresh).float()


class TernaryLinearTorch(nn.Module):
    """Ternary linear layer with STE for PyTorch."""

    def __init__(self, in_f: int, out_f: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        # Ternary weights are {-1,0,+1} with ~68% non-zero, so output
        # variance ≈ 0.68 * in_f * Var(x).  Scale gamma to normalise.
        self.gamma = nn.Parameter(torch.tensor(1.0 / math.sqrt(in_f)))

    @torch.autocast("cuda", enabled=False)
    @torch.autocast("cpu", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Force FP32 for ternary STE — quantization thresholds and
        # straight-through estimator must not be corrupted by FP16/BF16.
        x = x.float()
        w_q = ternary_quantize(self.weight)
        # STE: use quantised values in forward, real weights for grad
        w_ste = self.weight + (w_q - self.weight).detach()
        out = F.linear(x, w_ste, self.bias)
        return out * self.gamma


class TernaryExpertTorch(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = TernaryLinearTorch(d_model, d_ff, bias=False)
        self.up   = TernaryLinearTorch(d_model, d_ff, bias=False)
        self.down = TernaryLinearTorch(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DenseExpertTorch(nn.Module):
    """Standard SwiGLU FFN (used in dense_ffn ablation)."""
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoETorch(nn.Module):
    """Sparse MoE with configurable expert type (ternary or dense)."""

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        d_ff: int,
        use_ternary: bool = True,
        aux_loss_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        ExpertClass = TernaryExpertTorch if use_ternary else DenseExpertTorch
        self.experts = nn.ModuleList([ExpertClass(d_model, d_ff) for _ in range(n_experts)])
        self.router  = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)                              # (N, D)
        logits = self.router(x_flat)                           # (N, n_experts)
        probs  = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1) # (N, top_k)
        topk_weights = topk_probs / (topk_probs.sum(-1, keepdim=True) + 1e-9)

        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = (topk_idx[:, k] == e)
                if not mask.any():
                    continue
                w_e = topk_weights[mask, k:k+1]
                output[mask] = output[mask] + self.experts[e](x_flat[mask]) * w_e

        # Load-balancing auxiliary loss (negative entropy)
        mean_probs  = probs.mean(dim=0)
        entropy     = -(mean_probs * torch.log(mean_probs + 1e-9)).sum()
        aux_loss    = -entropy * self.aux_loss_coef

        return output.reshape(B, T, D), aux_loss


class MLAAttentionTorch(nn.Module):
    """Multi-Head Latent Attention (PyTorch) with SDPA support."""

    def __init__(self, d_model: int, num_heads: int, latent_dim: int) -> None:
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = self.head_dim ** -0.5
        self.latent_dim = latent_dim

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.kv_down  = nn.Linear(d_model, latent_dim, bias=False)
        self.k_up     = nn.Linear(latent_dim, d_model, bias=False)
        self.v_up     = nn.Linear(latent_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)
        latent = self.kv_down(x)    # (B, T, latent_dim)
        k = self.k_up(latent)
        v = self.v_up(latent)

        def split_heads(t):
            return t.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Use PyTorch 2.x scaled_dot_product_attention (auto-selects
        # FlashAttention-2 / MemoryEfficient kernel on CUDA)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class CausalSelfAttentionTorch(nn.Module):
    """Standard dense causal self-attention used for the transformer baseline."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


# Sequential SSM scan — numerically stable because
# decay = exp(A) ∈ (0,1) always shrinks the state, preventing overflow.
# JIT-scripted for performance (Python loop overhead is the main bottleneck).
# Gradient checkpointing must be DISABLED when using JIT — the recomputed
# tensor metadata differs from the forward pass, causing crashes.
@torch.jit.script
def _ssm_sequential_scan(
    x_in: torch.Tensor,   # (B, T, D) float32
    A: torch.Tensor,       # (B, T, N) float32, negative values
    B_ssm: torch.Tensor,   # (B, T, N) float32
    C_ssm: torch.Tensor,   # (B, T, N) float32
) -> torch.Tensor:
    B, T, D = x_in.shape
    N = A.shape[-1]
    device = x_in.device
    state = torch.zeros(B, D, N, device=device, dtype=x_in.dtype)
    outputs = torch.empty(B, T, D, device=device, dtype=x_in.dtype)
    for t in range(T):
        decay = torch.exp(A[:, t])                          # (B, N), values in (0, 1)
        state = state * decay.unsqueeze(1) + \
                x_in[:, t].unsqueeze(-1) * B_ssm[:, t].unsqueeze(1)  # (B, D, N)
        outputs[:, t] = (state * C_ssm[:, t].unsqueeze(1)).sum(-1)   # (B, D)
    return outputs


class MinimalSSMTorch(nn.Module):
    """Minimal SSM for PyTorch — sequential scan (numerically stable)."""

    def __init__(self, d_model: int, state_dim: int = 16) -> None:
        super().__init__()
        self.d_model   = d_model
        self.state_dim = state_dim
        d_inner = d_model * 2
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.A_proj   = nn.Linear(d_inner, state_dim, bias=False)
        self.B_proj   = nn.Linear(d_inner, state_dim, bias=False)
        self.C_proj   = nn.Linear(d_inner, state_dim, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.RMSNorm(d_inner)
        self.d_inner  = d_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        D = self.d_inner
        N = self.state_dim

        xz   = self.in_proj(x)
        x_in = F.silu(xz[..., :D]) * torch.sigmoid(xz[..., D:])

        # SSM parameters — all timesteps at once
        # Clamp A_proj to [-5, 0] before exp so A ∈ [-1, -0.0067]:
        #   decay = exp(A) ∈ (0.0067, 1.0) — always contractive.
        A = -torch.exp(self.A_proj(x_in).clamp(-5.0, 0.0))   # (B, T, N)
        B_ssm = self.B_proj(x_in)           # (B, T, N)
        C_ssm = self.C_proj(x_in)           # (B, T, N)

        # Sequential scan in float32 — stable because decay ∈ (0,1)
        out = _ssm_sequential_scan(
            x_in.float(), A.float(), B_ssm.float(), C_ssm.float(),
        )

        out = self.norm(out.to(x.dtype))
        return self.out_proj(out)


class IdentityBranch(nn.Module):
    """Ablation: replace a branch with a zero-output placeholder."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TransformerDenseLayerTorch(nn.Module):
    """Standard pre-norm Transformer block for the dense baseline."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttentionTorch(cfg.d_model, cfg.num_heads)
        self.ffn = DenseExpertTorch(cfg.d_model, cfg.d_ff)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        aux = torch.zeros((), device=x.device)
        return x, aux


def _make_ssm(cfg: TrainConfig) -> nn.Module:
    """Create SSM module based on config backend setting."""
    if cfg.ssm_backend == "mamba2":
        try:
            from mamba_ssm import Mamba2
            _log(f"Using Mamba-2 CUDA kernel (d_model={cfg.d_model}, d_state={cfg.state_dim})")
            return Mamba2(d_model=cfg.d_model, d_state=cfg.state_dim)
        except ImportError:
            _log("WARNING: mamba-ssm not installed, falling back to MinimalSSMTorch")
            return MinimalSSMTorch(cfg.d_model, cfg.state_dim)
    return MinimalSSMTorch(cfg.d_model, cfg.state_dim)


class TALHLayerTorch(nn.Module):
    """Single TALH hybrid layer (PyTorch version for training)."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.norm2 = nn.RMSNorm(cfg.d_model)

        ablation = cfg.ablation
        if ablation == "ssm_only":
            self.ssm = _make_ssm(cfg)
            self.mla = IdentityBranch(cfg.d_model)
        elif ablation == "mla_only":
            self.ssm = IdentityBranch(cfg.d_model)
            self.mla = MLAAttentionTorch(cfg.d_model, cfg.num_heads, cfg.latent_dim)
        elif ablation == "baseline":
            # Legacy baseline retained for compatibility with existing
            # 15M experiments. This is not a standard dense Transformer.
            self.ssm = IdentityBranch(cfg.d_model)
            self.mla = MLAAttentionTorch(cfg.d_model, cfg.num_heads, cfg.latent_dim)
        else:
            self.ssm = _make_ssm(cfg)
            self.mla = MLAAttentionTorch(cfg.d_model, cfg.num_heads, cfg.latent_dim)

        self.gate_proj = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False)

        use_ternary = (ablation not in {"dense_ffn", "baseline"})
        n_exp = 1 if ablation == "baseline" else cfg.n_experts
        top_k = 1 if ablation == "baseline" else cfg.top_k
        self.moe = MoETorch(
            cfg.d_model, n_exp, top_k, cfg.d_ff,
            use_ternary=use_ternary,
            aux_loss_coef=cfg.aux_loss_coef,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        h = self.norm1(x)

        if isinstance(self.ssm, IdentityBranch):
            h_ssm = self.ssm(h)
        else:
            h_ssm = self.ssm(h)

        if isinstance(self.mla, IdentityBranch):
            h_mla = self.mla(h)
        else:
            h_mla = self.mla(h, mask)

        h_cat  = torch.cat([h_ssm, h_mla], dim=-1)
        g      = torch.sigmoid(self.gate_proj(h_cat))
        h_fuse = g * h_mla + (1.0 - g) * h_ssm
        x      = residual + h_fuse

        h_moe, aux = self.moe(self.norm2(x))
        return x + h_moe, aux


class TALHTorch(nn.Module):
    """Full TALH language model (PyTorch)."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg   = cfg
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        LayerClass = TransformerDenseLayerTorch if cfg.ablation == "transformer_dense" else TALHLayerTorch
        self.layers = nn.ModuleList([LayerClass(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie embedding and lm_head weights
        self.lm_head.weight = self.embed.weight

        self._apply_base_init(init_std=0.02)

        # --- Near-zero init for residual-path output projections ---
        # This ensures the model starts as approximately identity-through-residual,
        # giving initial loss ≈ ln(vocab_size) ≈ 10.8 instead of exploding.
        # Scale factor decreases with depth (deeper layers contribute less initially).
        _log(f"Applying near-zero residual-path init for {cfg.n_layers} layers")
        with torch.no_grad():
            scale = (2 * cfg.n_layers) ** -0.5
            for layer in self.layers:
                if isinstance(layer, TransformerDenseLayerTorch):
                    layer.attn.out_proj.weight.mul_(scale)
                    layer.ffn.down.weight.mul_(scale)
                    continue
                # Gate fusion projection (merges SSM + MLA outputs)
                layer.gate_proj.weight.mul_(scale)
                # MLA output projection
                if hasattr(layer.mla, 'out_proj') and hasattr(layer.mla.out_proj, 'weight'):
                    layer.mla.out_proj.weight.mul_(scale)
                # SSM output projection
                if hasattr(layer.ssm, 'out_proj') and hasattr(layer.ssm.out_proj, 'weight'):
                    layer.ssm.out_proj.weight.mul_(scale)
                # MoE expert down projections (output back to residual stream)
                for expert in layer.moe.experts:
                    if hasattr(expert, 'down'):
                        down = expert.down
                        if isinstance(down, TernaryLinearTorch):
                            # Ternary quantization is relative to weight mean,
                            # so scaling weights has no effect on forward pass.
                            # Scale gamma instead to actually reduce output.
                            down.gamma.data.mul_(scale)
                        elif hasattr(down, 'weight'):
                            down.weight.mul_(scale)

    def _apply_base_init(self, init_std: float) -> None:
        """
        Use transformer-style initialisation before residual-path scaling.

        PyTorch's default Embedding init is N(0, 1), which is far too large
        for a tied LM head and causes the initial cross-entropy to explode.
        """
        _log(f"Applying transformer-style base init (std={init_std})")
        with torch.no_grad():
            nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)
            for module in self.modules():
                if module is self:
                    continue
                if isinstance(module, TernaryLinearTorch):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    module.gamma.fill_(1.0 / math.sqrt(module.weight.shape[1]))
                elif isinstance(module, nn.Linear):
                    if module is self.lm_head:
                        continue
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            # Identity branches must stay at exact zero for the ablations.
            for layer in self.layers:
                if isinstance(layer, TransformerDenseLayerTorch):
                    continue
                if isinstance(layer.ssm, IdentityBranch):
                    nn.init.zeros_(layer.ssm.proj.weight)
                if isinstance(layer.mla, IdentityBranch):
                    nn.init.zeros_(layer.mla.proj.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict:
        B, T = input_ids.shape
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        x   = self.embed(input_ids)
        aux = torch.zeros((), device=input_ids.device)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, layer_aux = torch.utils.checkpoint.checkpoint(
                    layer, x, mask, use_reentrant=False,
                )
            else:
                x, layer_aux = layer(x, mask)
            aux = aux + layer_aux

        logits = self.lm_head(self.norm(x))  # (B, T, vocab_size)

        result: dict = {"logits": logits, "aux_loss": aux}
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )
            result["ce_loss"]  = ce_loss
            result["loss"]     = ce_loss + aux
        return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenisedDataset(IterableDataset):
    """
    Streaming tokenised text dataset.

    Loads from HuggingFace datasets and tokenises on-the-fly with a
    GPT-2 / LLaMA-style tokeniser. For TinyStories the full dataset
    fits in memory; for FineWeb we stream.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        seq_len: int,
        tokenizer_name: str = "gpt2",
    ) -> None:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Suppress "sequence longer than model_max_length" warnings.
        # That limit is GPT-2's positional embedding constraint (1024 tokens),
        # which does not apply to TALH. We manually chunk into seq_len anyway.
        self.tok.model_max_length = 10 ** 9

        if dataset_name == "tinystories":
            self.ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        elif dataset_name == "fineweb":
            local_dir = os.environ.get("FINEWEB_LOCAL_DIR", "").strip()
            local_files = []
            if local_dir:
                local_files = sorted(
                    str(p) for p in Path(local_dir).rglob("*.parquet") if p.is_file()
                )

            if local_files:
                # Prefer locally cached parquet shards on remote GPUs. This avoids
                # repeated HF metadata/Xet round-trips during every training run.
                self.ds = load_dataset(
                    "parquet",
                    data_files=local_files,
                    split="train",
                    streaming=True,
                )
                if split == "validation":
                    self.ds = self.ds.skip(10_000)
            else:
                # FineWeb only has a "train" split. For validation we stream
                # from the same split but skip the first 500K documents so that
                # train and val data do not overlap in practice.
                actual_split = "train"
                self.ds = load_dataset(
                    "HuggingFaceFW/fineweb", name="sample-10BT",
                    split=actual_split, streaming=True,
                )
                if split == "validation":
                    self.ds = self.ds.skip(500_000)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def __iter__(self) -> Iterator[dict]:
        ds = self.ds
        worker = get_worker_info()
        if worker is not None and hasattr(ds, "shard"):
            ds = ds.shard(num_shards=worker.num_workers, index=worker.id)
        buf: list[int] = []
        for sample in ds:
            text = sample.get("text") or sample.get("story") or ""
            ids  = self.tok.encode(text, add_special_tokens=True)
            buf.extend(ids)
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf   = buf[self.seq_len:]
                x = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
                y = torch.tensor(chunk[1:self.seq_len + 1], dtype=torch.long)
                yield {"input_ids": x, "labels": y}


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine schedule with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# CSV Training Log
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "step", "train_loss", "aux_loss", "val_loss", "lr",
    "elapsed_sec", "seq_len", "gpu_mem_gb",
]


class CSVLogger:
    """Append-only CSV logger. Crash-resilient (flushes after each write)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_FIELDS)
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def log(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in _CSV_FIELDS})
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _rotate_checkpoints(out_dir: Path, keep_last_n: int) -> None:
    """Keep only the last N step checkpoints + best.pt."""
    step_files = sorted(out_dir.glob("step_*.pt"))
    if len(step_files) > keep_last_n:
        for old_ckpt in step_files[:-keep_last_n]:
            old_ckpt.unlink()
            _log(f"Removed old checkpoint: {old_ckpt.name}")


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP and/or torch.compile wrappers to get the raw model."""
    if hasattr(model, 'module'):      # DDP
        model = model.module
    if hasattr(model, '_orig_mod'):   # torch.compile
        model = model._orig_mod
    return model


def _save_checkpoint(
    out_dir: Path,
    filename: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
    best_val_loss: float,
    is_ddp: bool,
) -> None:
    """Save checkpoint with model + optimizer state for full resumability."""
    state_dict = _unwrap_model(model).state_dict()
    torch.save(
        {
            "step": step,
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "best_val_loss": best_val_loss,
        },
        out_dir / filename,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    """Main training loop with AMP, gradient accumulation, and DDP support."""
    torch.manual_seed(cfg.seed)

    # --- Device setup (DDP or single GPU/MPS/CPU) ---
    is_ddp = cfg.use_ddp and torch.cuda.is_available()
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    is_main = not is_ddp or int(os.environ.get("LOCAL_RANK", 0)) == 0

    if is_main:
        _log(f"Training on device: {device}")
        _log(f"Ablation variant: {cfg.ablation}")
        _log(f"AMP: {cfg.use_amp} | Grad Accum: {cfg.grad_accumulation_steps} "
             f"| Grad Ckpt: {cfg.gradient_checkpointing} | SSM: {cfg.ssm_backend}")
        _log(
            f"Curriculum: {cfg.curriculum_steps} | Val batches: {cfg.val_batches} "
            f"| Workers train/val: {cfg.num_workers}/{cfg.val_num_workers}"
        )

    # Save config
    out_dir = Path(cfg.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    model = TALHTorch(cfg).to(device)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
        )

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        _log(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # --- AMP setup ---
    amp_enabled = cfg.use_amp and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    # --- CSV logger ---
    csv_logger = None
    if cfg.log_csv and is_main:
        csv_logger = CSVLogger(cfg.log_csv)

    # --- WandB ---
    wandb_run = None
    if cfg.wandb_project and is_main:
        try:
            import hashlib
            import wandb
            # Build a deterministic run ID from project + ablation + model shape
            # so that checkpoint-based restarts resume the SAME WandB run
            # instead of creating duplicates with identical names.
            id_seed = f"{cfg.wandb_project}/{cfg.ablation}_{cfg.d_model}d_{cfg.n_layers}L"
            wandb_id = hashlib.sha256(id_seed.encode()).hexdigest()[:12]
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                id=wandb_id,
                name=f"{cfg.ablation}_{cfg.d_model}d_{cfg.n_layers}L",
                config=asdict(cfg),
                resume="allow",
            )
        except ImportError:
            _log("WARNING: wandb not installed, skipping experiment tracking")

    # --- Dataset ---
    current_seq_len = cfg.curriculum_steps[0][1]
    dataset = TokenisedDataset(cfg.dataset, split="train", seq_len=current_seq_len)
    loader  = DataLoader(
        dataset, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, pin_memory=device.startswith("cuda"),
    )
    data_iter = iter(loader)

    def _make_val_loader(seq_len: int) -> DataLoader:
        vds = TokenisedDataset(cfg.dataset, split="validation", seq_len=seq_len)
        return DataLoader(
            vds, batch_size=cfg.batch_size,
            num_workers=cfg.val_num_workers, pin_memory=device.startswith("cuda"),
        )

    best_val_loss = float("inf")
    start_step = 0

    # --- Resume from checkpoint ---
    _existing = sorted(out_dir.glob("step_*.pt"))
    _resume_path = _existing[-1] if _existing else (out_dir / "best.pt")
    if _resume_path.exists():
        _ckpt = torch.load(str(_resume_path), map_location=device, weights_only=False)
        raw_model = _unwrap_model(model)
        raw_model.load_state_dict(_ckpt["model"])
        if "optimizer" in _ckpt:
            optimizer.load_state_dict(_ckpt["optimizer"])
        start_step = _ckpt.get("step", 0)
        best_val_loss = _ckpt.get("best_val_loss", best_val_loss)
        if is_main:
            _log(f"Resuming from {_resume_path.name} at step {start_step} "
                 f"(best_val_loss={best_val_loss:.4f})")
        del _ckpt

    t0 = time.perf_counter()

    for step in range(start_step + 1, cfg.max_steps + 1):
        # Context length curriculum
        for cutoff, sl in reversed(cfg.curriculum_steps):
            if step >= cutoff:
                if sl != current_seq_len:
                    if is_main:
                        _log(f"Step {step}: extending context to {sl}")
                    current_seq_len = sl
                    dataset  = TokenisedDataset(cfg.dataset, split="train", seq_len=sl)
                    loader   = DataLoader(
                        dataset, batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers, pin_memory=device.startswith("cuda"),
                    )
                    data_iter = iter(loader)
                break

        # Learning rate update
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(step, cfg)

        # --- Forward + backward with gradient accumulation ---
        model.train()
        optimizer.zero_grad()
        accum_ce_loss = 0.0
        accum_aux_loss = 0.0

        for _micro in range(cfg.grad_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with torch.autocast(
                device_type=device.split(":")[0],
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                out  = model(input_ids, targets=labels)
                loss = out["loss"] / cfg.grad_accumulation_steps

            scaler.scale(loss).backward()
            accum_ce_loss  += out["ce_loss"].item() / cfg.grad_accumulation_steps
            accum_aux_loss += out["aux_loss"].item() / cfg.grad_accumulation_steps

        # NaN-safe: skip optimizer step if loss is NaN/Inf
        if not (math.isfinite(accum_ce_loss) and math.isfinite(accum_aux_loss)):
            if is_main and step % 100 == 0:
                _log(f"step={step:6d} | NaN/Inf detected — skipping optimizer step")
            optimizer.zero_grad()
            scaler.update()
            continue

        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # --- Logging ---
        if step % cfg.log_every == 0 and is_main:
            elapsed = time.perf_counter() - t0
            gpu_mem_gb = (
                torch.cuda.max_memory_allocated() / 1e9
                if device.startswith("cuda") else 0.0
            )
            _log(
                f"step={step:6d} | loss={accum_ce_loss:.4f} "
                f"| aux={accum_aux_loss:.4f} "
                f"| gnorm={grad_norm:.2f} "
                f"| lr={get_lr(step, cfg):.2e} "
                f"| mem={gpu_mem_gb:.1f}GB "
                f"| elapsed={elapsed:.1f}s"
            )
            if wandb_run:
                wandb_run.log({
                    "train/ce_loss": accum_ce_loss,
                    "train/aux_loss": accum_aux_loss,
                    "train/grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                    "train/lr": get_lr(step, cfg),
                    "train/gpu_mem_gb": gpu_mem_gb,
                    "train/seq_len": current_seq_len,
                }, step=step)

        # --- Validation ---
        if step % cfg.eval_every == 0 and is_main:
            val_loss = _evaluate(model, _make_val_loader(current_seq_len),
                                 device, n_batches=cfg.val_batches)
            _log(f"step={step} | val_loss={val_loss:.4f}")

            if wandb_run:
                wandb_run.log({"val/loss": val_loss}, step=step)

            if csv_logger:
                elapsed = time.perf_counter() - t0
                gpu_mem_gb = (
                    torch.cuda.max_memory_allocated() / 1e9
                    if device.startswith("cuda") else 0.0
                )
                csv_logger.log({
                    "step": step,
                    "train_loss": round(accum_ce_loss, 6),
                    "aux_loss": round(accum_aux_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "lr": f"{get_lr(step, cfg):.2e}",
                    "elapsed_sec": round(elapsed, 1),
                    "seq_len": current_seq_len,
                    "gpu_mem_gb": round(gpu_mem_gb, 2),
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(
                    out_dir, "best.pt", model, optimizer, cfg,
                    step, best_val_loss, is_ddp,
                )

        # --- Step checkpoint ---
        if step % cfg.save_every == 0 and is_main:
            _save_checkpoint(
                out_dir, f"step_{step:06d}.pt", model, optimizer, cfg,
                step, best_val_loss, is_ddp,
            )
            _rotate_checkpoints(out_dir, cfg.keep_last_n)

    if is_main:
        _log("Training complete.")
        _log(f"Best validation loss: {best_val_loss:.4f}")
        if csv_logger:
            csv_logger.close()
        if wandb_run:
            wandb_run.finish()

    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_batches: int = 50,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            out = model(input_ids, targets=labels)
            total_loss += out["ce_loss"].item()
            n += 1
    return total_loss / max(1, n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train TALH model")
    # Model
    p.add_argument("--dataset",       default="tinystories")
    p.add_argument("--vocab_size",    type=int,   default=32000)
    p.add_argument("--n_layers",      type=int,   default=4)
    p.add_argument("--d_model",       type=int,   default=256)
    p.add_argument("--num_heads",     type=int,   default=4)
    p.add_argument("--latent_dim",    type=int,   default=64)
    p.add_argument("--state_dim",     type=int,   default=16)
    p.add_argument("--n_experts",     type=int,   default=8)
    p.add_argument("--top_k",         type=int,   default=2)
    p.add_argument("--d_ff",          type=int,   default=512)
    p.add_argument("--ablation",      default="full",
                   choices=["full", "no_ttt", "mla_only", "ssm_only", "dense_ffn", "baseline", "transformer_dense"])
    # Training
    p.add_argument("--max_steps",     type=int,   default=50_000)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup_steps",  type=int,   default=1_000)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--eval_every",    type=int,   default=500)
    p.add_argument("--save_every",    type=int,   default=5_000)
    p.add_argument("--log_every",     type=int,   default=100)
    p.add_argument("--val_batches",   type=int,   default=50)
    p.add_argument(
        "--curriculum",
        default="",
        help="Comma-separated step:seq_len pairs, e.g. 0:256,3000:512",
    )
    # Scaling
    p.add_argument("--grad_accumulation_steps", type=int, default=1)
    p.add_argument("--use_amp",       action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_ddp",       action="store_true")
    p.add_argument("--ssm_backend",   default="minimal", choices=["minimal", "mamba2"])
    p.add_argument("--num_workers",   type=int,   default=-1)
    p.add_argument("--val_num_workers", type=int, default=-1)
    # Logging
    p.add_argument("--wandb_project", default="")
    p.add_argument("--log_csv",       default="")
    # I/O
    p.add_argument("--output_dir",    default="./checkpoints/talh")
    p.add_argument("--seed",          type=int,   default=42)

    args = p.parse_args()
    cfg_kwargs = {k: v for k, v in vars(args).items() if k != "curriculum" and v is not None}
    cfg = TrainConfig(**cfg_kwargs)
    if args.curriculum:
        cfg.curriculum_steps = _parse_curriculum_spec(args.curriculum)
    # Append ablation to output dir if not already present
    if not cfg.output_dir.endswith(f"_{args.ablation}"):
        cfg.output_dir = f"{cfg.output_dir}_{args.ablation}"
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
