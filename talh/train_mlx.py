"""
TALH Training Script — MLX / Mac (Apple Silicon).

Trains a small TALH model on TinyStories using MLX's native optimizer
and functional gradient API. Designed to run locally on Apple Silicon
without needing Colab or a GPU.

Expected training time on M-series Mac (~15M params, TinyStories):
  M1/M2 (16GB): ~4–6 hrs for 50k steps
  M3/M4 (32GB+): ~2–3 hrs for 50k steps
  Reduce --max_steps to 5000–10000 for a quick validation run.

Usage:
    # Quick validation (~10 min, checks model trains and loss decreases)
    python -m talh.train_mlx --max_steps 2000 --output_dir checkpoints/talh_quick

    # Full training run
    python -m talh.train_mlx --max_steps 50000 --output_dir checkpoints/talh_full

    # Ablation variant (mla_only / ssm_only / dense_ffn / baseline)
    python -m talh.train_mlx --ablation mla_only --output_dir checkpoints/talh_mla_only
"""

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model
    vocab_size:   int   = 4096    # GPT-2 tokeniser (gpt2 has 50257, we use small vocab for speed)
    n_layers:     int   = 4
    d_model:      int   = 256
    num_heads:    int   = 4
    latent_dim:   int   = 64
    state_dim:    int   = 16
    n_experts:    int   = 8
    top_k:        int   = 2
    d_ff:         int   = 512
    fusion:       str   = "gated"
    ablation:     str   = "full"

    # Training
    dataset:      str   = "tinystories"
    seq_len:      int   = 256
    batch_size:   int   = 8
    max_steps:    int   = 50_000
    eval_every:   int   = 500
    save_every:   int   = 5_000
    lr:           float = 3e-4
    weight_decay: float = 0.1
    grad_clip:    float = 1.0
    warmup_steps: int   = 500
    aux_loss_coef:float = 0.01

    # Curriculum: (step, seq_len) pairs
    curriculum: list = None

    # I/O
    output_dir:   str   = "./checkpoints/talh"
    seed:         int   = 42

    def __post_init__(self):
        if self.curriculum is None:
            self.curriculum = [
                (0,      128),
                (5_000,  256),
                (15_000, 512),
                (30_000, 1024),
            ]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_tinystories(split: str = "train"):
    """Load TinyStories from HuggingFace datasets."""
    from datasets import load_dataset
    logger.info(f"Loading TinyStories ({split})...")
    return load_dataset("roneneldan/TinyStories", split=split, streaming=True)


def simple_tokenise(text: str, vocab_size: int) -> list[int]:
    """
    Character-level tokeniser (no external dependency).
    Maps each char to an id in [1, vocab_size-1].
    Fast to run on Mac without downloading a full tokeniser.
    """
    return [max(1, ord(c) % vocab_size) for c in text]


def get_batch(
    stories,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    _buf: list,
) -> tuple[mx.array, mx.array]:
    """
    Fill a batch from the story stream, buffering tokens across calls.

    Args:
        stories: HuggingFace streaming dataset iterator.
        seq_len: Tokens per sequence.
        batch_size: Number of sequences per batch.
        vocab_size: Vocab size for tokenisation.
        _buf: Mutable token buffer (shared across calls, modified in place).

    Returns:
        (input_ids, labels) each of shape (batch_size, seq_len).
    """
    needed = batch_size * (seq_len + 1)
    while len(_buf) < needed:
        try:
            item = next(stories)
            text = item.get("story") or item.get("text") or ""
            _buf.extend(simple_tokenise(text + "\n", vocab_size))
        except StopIteration:
            break

    xs, ys = [], []
    for i in range(batch_size):
        start = i * (seq_len + 1)
        chunk = _buf[start:start + seq_len + 1]
        if len(chunk) < seq_len + 1:
            chunk += [0] * (seq_len + 1 - len(chunk))
        xs.append(chunk[:seq_len])
        ys.append(chunk[1:seq_len + 1])

    # Remove consumed tokens
    del _buf[:batch_size * (seq_len + 1)]

    x = mx.array(xs, dtype=mx.uint32)
    y = mx.array(ys, dtype=mx.uint32)
    return x, y


# ---------------------------------------------------------------------------
# Model with ablation support (wraps talh.model.TALH)
# ---------------------------------------------------------------------------

def build_model(cfg: TrainConfig):
    """Build TALH model based on ablation variant."""
    from talh.model import TALH, TALHConfig

    model_cfg = TALHConfig(
        vocab_size=cfg.vocab_size,
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        latent_dim=cfg.latent_dim,
        state_dim=cfg.state_dim,
        n_experts=cfg.n_experts if cfg.ablation not in {"baseline"} else 1,
        top_k=cfg.top_k if cfg.ablation not in {"baseline"} else 1,
        d_ff=cfg.d_ff,
        fusion=cfg.fusion,
        max_seq_len=max(sl for _, sl in cfg.curriculum),
    )

    model = TALH(model_cfg)

    # Patch SSM/MLA branches for ablation variants
    if cfg.ablation == "ssm_only":
        _disable_mla_branches(model)
    elif cfg.ablation == "mla_only":
        _disable_ssm_branches(model)
    elif cfg.ablation in {"dense_ffn", "baseline"}:
        _replace_moe_with_dense(model, cfg)

    return model


def _disable_mla_branches(model):
    """Zero-out MLA output projections (ablation: SSM-only)."""
    for layer in model.layers:
        layer.mla.out_proj.weight = mx.zeros_like(layer.mla.out_proj.weight)


def _disable_ssm_branches(model):
    """Zero-out SSM output projections (ablation: MLA-only)."""
    for layer in model.layers:
        layer.ssm.core.out_proj.weight = mx.zeros_like(layer.ssm.core.out_proj.weight)


def _replace_moe_with_dense(model, cfg: TrainConfig):
    """Replace ternary MoE with single dense expert (ablation: dense_ffn/baseline)."""
    from talh.layers.ternary_moe import TernaryMoE
    for i, layer in enumerate(model.layers):
        layer.moe = TernaryMoE(
            d_model=cfg.d_model,
            n_experts=1,
            top_k=1,
            d_ff=cfg.d_ff,
        )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def loss_fn(model, x: mx.array, y: mx.array) -> mx.array:
    """Cross-entropy LM loss + MoE auxiliary loss."""
    logits, _, _, aux_loss = model(x)
    # Cross-entropy over all positions
    B, T, V = logits.shape
    ce = nn.losses.cross_entropy(
        logits.reshape(B * T, V),
        y.reshape(B * T),
    ).mean()
    return ce + aux_loss, ce


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine schedule with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg: TrainConfig) -> None:
    mx.random.seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    model = build_model(cfg)
    # Count parameters by recursively walking the nested param dict
    def _count(p):
        if isinstance(p, mx.array): return p.size
        if isinstance(p, dict): return sum(_count(v) for v in p.values())
        if isinstance(p, list): return sum(_count(v) for v in p)
        return 0
    n_params = _count(model.trainable_parameters())
    logger.info(f"Model parameters: {n_params:,}  |  ablation: {cfg.ablation}")

    optimizer = optim.AdamW(
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # Compile value_and_grad for the loss
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Load dataset
    train_stories = iter(load_tinystories("train"))
    val_stories   = iter(load_tinystories("validation"))
    train_buf: list[int] = []
    val_buf:   list[int] = []

    current_seq_len = cfg.curriculum[0][1]
    best_val_loss = float("inf")
    t0 = time.perf_counter()

    def _flat_params(params, prefix=""):
        """Flatten nested parameter dict to {dotted.key: array}."""
        flat = {}
        if isinstance(params, mx.array):
            flat[prefix] = params
        elif isinstance(params, dict):
            for k, v in params.items():
                flat.update(_flat_params(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(params, list):
            for i, v in enumerate(params):
                flat.update(_flat_params(v, f"{prefix}.{i}" if prefix else str(i)))
        return flat

    for step in range(1, cfg.max_steps + 1):
        # Curriculum: update seq_len if needed
        for cutoff, sl in reversed(cfg.curriculum):
            if step >= cutoff:
                if sl != current_seq_len:
                    logger.info(f"Step {step}: context length → {sl}")
                    current_seq_len = sl
                break

        # Learning rate
        lr = get_lr(step, cfg)
        optimizer.learning_rate = lr

        # Batch
        x, y = get_batch(train_stories, current_seq_len, cfg.batch_size,
                         cfg.vocab_size, train_buf)

        # Forward + backward
        (loss, ce_loss), grads = loss_and_grad(model, x, y)

        # Gradient clipping — walk nested dict structure
        def _collect_leaves(p):
            if isinstance(p, mx.array): return [p]
            if isinstance(p, dict): return [l for v in p.values() for l in _collect_leaves(v)]
            if isinstance(p, list): return [l for v in p for l in _collect_leaves(v)]
            return []

        def _scale_leaves(p, s):
            if isinstance(p, mx.array): return p * s
            if isinstance(p, dict): return {k: _scale_leaves(v, s) for k, v in p.items()}
            if isinstance(p, list): return [_scale_leaves(v, s) for v in p]
            return p

        leaves = _collect_leaves(grads)
        if leaves:
            grad_norm = math.sqrt(sum(mx.sum(g ** 2).item() for g in leaves))
            if grad_norm > cfg.grad_clip:
                grads = _scale_leaves(grads, cfg.grad_clip / (grad_norm + 1e-6))

        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), loss)

        # Logging
        if step % 100 == 0:
            elapsed = time.perf_counter() - t0
            logger.info(
                f"step={step:6d} | loss={ce_loss.item():.4f} "
                f"| lr={lr:.2e} | seq_len={current_seq_len} "
                f"| elapsed={elapsed:.0f}s"
            )

        # Validation
        if step % cfg.eval_every == 0:
            val_loss = _eval(model, val_stories, val_buf, current_seq_len,
                             cfg.batch_size, cfg.vocab_size, n_batches=20)
            logger.info(f"step={step} | val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mx.save_safetensors(
                    str(out_dir / "best.safetensors"),
                    _flat_params(model.trainable_parameters()),
                )
                logger.info(f"  → saved best (val_loss={val_loss:.4f})")

        # Checkpoint
        if step % cfg.save_every == 0:
            mx.save_safetensors(
                str(out_dir / f"step_{step:06d}.safetensors"),
                _flat_params(model.trainable_parameters()),
            )

    logger.info(f"Done. Best val_loss: {best_val_loss:.4f}")


def _eval(model, stories, buf, seq_len, batch_size, vocab_size, n_batches=20):
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(stories, seq_len, batch_size, vocab_size, buf)
        logits, _, _, _ = model(x)
        B, T, V = logits.shape
        ce = nn.losses.cross_entropy(
            logits.reshape(B * T, V), y.reshape(B * T)
        ).mean()
        mx.eval(ce)
        total += ce.item()
    return total / n_batches


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--n_layers",   type=int,   default=4)
    p.add_argument("--d_model",    type=int,   default=256)
    p.add_argument("--n_experts",  type=int,   default=8)
    p.add_argument("--top_k",      type=int,   default=2)
    p.add_argument("--d_ff",       type=int,   default=512)
    p.add_argument("--max_steps",  type=int,   default=50_000)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--eval_every", type=int,   default=500)
    p.add_argument("--save_every", type=int,   default=5_000)
    p.add_argument("--ablation",   default="full",
                   choices=["full", "mla_only", "ssm_only", "dense_ffn", "baseline"])
    p.add_argument("--output_dir", default="./checkpoints/talh")
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()
    cfg = TrainConfig(**{k: v for k, v in vars(args).items()})
    cfg.output_dir = f"{args.output_dir}_{args.ablation}"
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
