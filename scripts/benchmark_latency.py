"""
H2 — Latency Benchmark: TTFT & TPOT (Mac / MLX).

Measures Time To First Token (TTFT) and Time Per Output Token (TPOT)
for a TALH model at different prompt lengths.

Compares against an mlx-lm baseline model (e.g. Qwen2.5-0.5B or SmolLM).

Usage:
    # Benchmark TALH (randomly-initialised, for structural comparison)
    python scripts/benchmark_latency.py \
        --model talh \
        --prompt_lens 256 1024 4096 \
        --n_tokens 50 \
        --output_dir results/

    # Benchmark a real mlx-lm model as baseline
    python scripts/benchmark_latency.py \
        --model mlx_baseline \
        --mlx_model_path mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --prompt_lens 256 1024 4096 \
        --output_dir results/
"""

import argparse
import csv
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from talh.model import TALH, TALHConfig


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync() -> None:
    """Force MLX lazy evaluation to complete."""
    # Create a trivial array and eval it to flush the graph
    x = mx.array([0.0])
    mx.eval(x)


def measure_ttft(
    model_fn,
    input_ids: mx.array,
) -> float:
    """
    Measure Time To First Token: time from prompt to first output token.

    Args:
        model_fn: Callable that takes input_ids and returns (logits, ...)
        input_ids: Prompt token ids, shape (1, prompt_len).

    Returns:
        TTFT in milliseconds.
    """
    _sync()
    t0 = time.perf_counter()
    out = model_fn(input_ids)
    if isinstance(out, tuple):
        mx.eval(out[0])          # logits
    else:
        mx.eval(out)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def measure_tpot(
    model_fn,
    prompt_ids: mx.array,
    n_tokens: int = 50,
) -> float:
    """
    Measure Time Per Output Token (decode throughput).

    Runs the model for n_tokens decode steps (single-token at a time,
    simulating autoregressive generation) and returns mean ms/token.

    Args:
        model_fn:   Callable(input_ids) → (logits, *cache_stuff)
        prompt_ids: Prompt of shape (1, prompt_len).
        n_tokens:   Number of decode tokens to generate.

    Returns:
        Mean TPOT in milliseconds.
    """
    # Prefill
    out = model_fn(prompt_ids)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    mx.eval(logits)

    # Get next token
    next_id = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    # Decode loop
    latencies = []
    for _ in range(n_tokens):
        _sync()
        t0 = time.perf_counter()
        out = model_fn(next_id)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        mx.eval(logits)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        next_id = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    return float(np.mean(latencies))


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

def make_talh_fn(cfg: TALHConfig):
    """Return a simple callable for TALH forward (ignores caches)."""
    model = TALH(cfg)

    def forward(input_ids: mx.array):
        logits, _, _, _ = model(input_ids)
        return logits, None

    return forward


def make_mlx_baseline_fn(model_path: str):
    """Load an mlx-lm model and return a forward callable."""
    try:
        from mlx_lm import load
        model, tokenizer = load(model_path)

        def forward(input_ids: mx.array):
            logits = model(input_ids)
            return logits, None

        return forward
    except ImportError:
        raise ImportError("mlx_lm is not installed. Run: pip install mlx-lm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TALHConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
    )

    if args.model == "talh":
        model_fn = make_talh_fn(cfg)
        model_name = "talh_random_init"
    elif args.model == "mlx_baseline":
        if not args.mlx_model_path:
            raise ValueError("--mlx_model_path required for mlx_baseline")
        model_fn = make_mlx_baseline_fn(args.mlx_model_path)
        model_name = Path(args.mlx_model_path).name
    else:
        raise ValueError(f"Unknown --model: {args.model}")

    print(f"\n=== Latency Benchmark: {model_name} ===")
    print(f"{'Prompt Len':>12} | {'TTFT (ms)':>12} | {'TPOT (ms)':>12} | {'Tokens/s':>10}")
    print("-" * 56)

    rows = []
    for plen in args.prompt_lens:
        input_ids = mx.zeros((1, plen), dtype=mx.uint32)

        # Warmup
        _ = measure_ttft(model_fn, input_ids)

        # Measure
        ttft = measure_ttft(model_fn, input_ids)
        tpot = measure_tpot(model_fn, input_ids, n_tokens=args.n_tokens)
        toks_per_s = 1000.0 / tpot if tpot > 0 else 0

        print(f"{plen:>12,} | {ttft:>12.1f} | {tpot:>12.2f} | {toks_per_s:>10.1f}")
        rows.append({
            "model": model_name,
            "prompt_len": plen,
            "ttft_ms": round(ttft, 2),
            "tpot_ms": round(tpot, 3),
            "tokens_per_sec": round(toks_per_s, 1),
            "n_decode_tokens": args.n_tokens,
        })

    csv_path = out_dir / f"latency_{model_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H2: TTFT/TPOT latency benchmark")
    p.add_argument("--model",           default="talh", choices=["talh", "mlx_baseline"])
    p.add_argument("--mlx_model_path",  default=None)
    p.add_argument("--prompt_lens",     nargs="+", type=int, default=[256, 512, 1024, 4096])
    p.add_argument("--n_tokens",        type=int, default=50)
    p.add_argument("--n_layers",        type=int, default=4)
    p.add_argument("--d_model",         type=int, default=256)
    p.add_argument("--num_heads",       type=int, default=4)
    p.add_argument("--latent_dim",      type=int, default=64)
    p.add_argument("--output_dir",      default="results/")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
