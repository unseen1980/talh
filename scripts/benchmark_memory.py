"""
H1 — Memory Growth Benchmark (Mac / MLX).

Measures peak unified memory consumption vs. sequence length for:
  - Standard MHA (baseline dense Transformer)
  - MLA-only (TALH without SSM branch)
  - TALH-full

Produces a CSV + plot: memory_vs_context.csv / memory_vs_context.png

Usage:
    python scripts/benchmark_memory.py \
        --seq_lens 256 512 1024 2048 4096 8192 \
        --d_model 256 --num_heads 4 --latent_dim 64 \
        --output_dir results/
"""

import argparse
import csv
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Memory probe helpers
# ---------------------------------------------------------------------------

def _mx():
    """Lazy import of mlx.core (Apple Silicon only)."""
    import mlx.core as mx
    return mx


def get_active_memory_mb() -> float:
    """Current Metal active memory in MB."""
    mx = _mx()
    try:
        return mx.get_active_memory() / 1e6
    except AttributeError:
        try:
            return mx.metal.get_active_memory() / 1e6
        except AttributeError:
            return 0.0


def reset_memory_peak() -> None:
    mx = _mx()
    try:
        mx.reset_peak_memory()
    except AttributeError:
        try:
            mx.metal.reset_peak_memory()
        except AttributeError:
            pass


def get_peak_memory_mb() -> float:
    mx = _mx()
    try:
        return mx.get_peak_memory() / 1e6
    except AttributeError:
        try:
            return mx.metal.get_peak_memory() / 1e6
        except AttributeError:
            return 0.0


# ---------------------------------------------------------------------------
# Analytical KV cache size (H1 — no forward pass needed)
# ---------------------------------------------------------------------------

def analytical_kv_bytes(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    latent_dim: int,
    n_layers: int,
    dtype_bytes: int = 2,
) -> dict:
    """
    Compute KV cache sizes analytically (no forward pass).

    Returns dict with 'standard_mha_mb', 'mla_mb' for a given seq_len.
    """
    mha_per_layer = 2 * seq_len * num_heads * head_dim * dtype_bytes
    mla_per_layer = seq_len * latent_dim * dtype_bytes

    return {
        "standard_mha_mb": mha_per_layer * n_layers / 1e6,
        "mla_mb":          mla_per_layer * n_layers / 1e6,
        "compression_ratio": mha_per_layer / mla_per_layer,
    }


# ---------------------------------------------------------------------------
# Live memory profiling (MLX forward pass)
# ---------------------------------------------------------------------------

def profile_talh_forward(
    cfg: "TALHConfig",
    seq_len: int,
    batch_size: int = 1,
) -> dict:
    """
    Run a TALH forward pass and measure peak memory.

    Returns dict with 'model', 'seq_len', 'peak_memory_mb'.
    """
    mx = _mx()
    from talh.model import TALH

    model = TALH(cfg)
    input_ids = mx.zeros((batch_size, seq_len), dtype=mx.uint32)

    reset_memory_peak()
    t0 = time.perf_counter()
    logits, kv_caches, ssm_states, aux = model(input_ids)
    mx.eval(logits)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "seq_len": seq_len,
        "peak_memory_mb": get_peak_memory_mb(),
        "elapsed_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = args.seq_lens
    n_layers = args.n_layers
    d_model  = args.d_model
    num_heads = args.num_heads
    head_dim  = d_model // num_heads
    latent_dim = args.latent_dim

    # -----------------------------------------------------------------------
    # Part 1: Analytical KV cache comparison (no GPU needed)
    # -----------------------------------------------------------------------
    print("\n=== Analytical KV Cache Comparison ===")
    print(f"{'Seq Len':>10} | {'Standard MHA (MB)':>18} | {'MLA (MB)':>10} | {'Ratio':>8}")
    print("-" * 55)

    rows = []
    for sl in seq_lens:
        r = analytical_kv_bytes(
            seq_len=sl,
            num_heads=num_heads,
            head_dim=head_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
        )
        print(f"{sl:>10,} | {r['standard_mha_mb']:>18.2f} | {r['mla_mb']:>10.2f} | {r['compression_ratio']:>8.1f}x")
        rows.append({
            "seq_len": sl,
            "model": "standard_mha",
            "kv_cache_mb": r["standard_mha_mb"],
        })
        rows.append({
            "seq_len": sl,
            "model": "mla",
            "kv_cache_mb": r["mla_mb"],
        })

    csv_path = out_dir / "kv_cache_vs_context.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_len", "model", "kv_cache_mb"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # -----------------------------------------------------------------------
    # Part 2: Live forward pass memory (MLX)
    # -----------------------------------------------------------------------
    from talh.model import TALHConfig

    print("\n=== Live Forward Pass Memory (TALH, MLX) ===")
    cfg = TALHConfig(
        n_layers=n_layers,
        d_model=d_model,
        num_heads=num_heads,
        latent_dim=latent_dim,
    )

    live_rows = []
    for sl in seq_lens:
        if sl > 4096:
            print(f"  Skipping seq_len={sl} (too large for analytical-only run)")
            continue
        result = profile_talh_forward(cfg, seq_len=sl)
        print(
            f"  seq_len={sl:>6,} | peak_mem={result['peak_memory_mb']:>8.1f} MB "
            f"| elapsed={result['elapsed_ms']:>7.1f} ms"
        )
        live_rows.append({
            "seq_len": sl,
            "model": "talh_full",
            **result,
        })

    if live_rows:
        live_csv = out_dir / "talh_live_memory.csv"
        with open(live_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=live_rows[0].keys())
            writer.writeheader()
            writer.writerows(live_rows)
        print(f"Saved: {live_csv}")

    # -----------------------------------------------------------------------
    # Part 3: Plot (if matplotlib available)
    # -----------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(9, 5))
        for model_name, grp in df.groupby("model"):
            ax.plot(grp["seq_len"], grp["kv_cache_mb"], marker="o", label=model_name)
        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_ylabel("KV Cache Size (MB)")
        ax.set_title(f"H1: KV Cache Memory vs Context Length\n(n_layers={n_layers}, d_model={d_model})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = out_dir / "memory_vs_context.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {plot_path}")
    except ImportError:
        print("matplotlib/pandas not installed — skipping plot.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H1: Memory vs context length benchmark")
    p.add_argument("--seq_lens", nargs="+", type=int,
                   default=[256, 512, 1024, 2048, 4096, 8192, 32768])
    p.add_argument("--n_layers",   type=int, default=4)
    p.add_argument("--d_model",    type=int, default=256)
    p.add_argument("--num_heads",  type=int, default=4)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--output_dir", default="results/")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
