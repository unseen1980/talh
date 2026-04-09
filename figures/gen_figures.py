"""
Generate publication figures for the TALH 0.5B paper.
Run from the paper/figures/ directory or adjust OUTPUT_DIR.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Shared style ─────────────────────────────────────────────────────────────
COLORS = {
    "dense_ffn":        "#2196F3",   # blue  – the winner
    "full":             "#FF9800",   # orange
    "ssm_only":         "#9C27B0",   # purple
    "transformer_dense":"#607D8B",   # grey
    "mla_only":         "#F44336",   # red
}
LABELS = {
    "dense_ffn":        "TALH dense-FFN",
    "full":             "TALH full (MoE)",
    "ssm_only":         "SSM only",
    "transformer_dense":"Dense Transformer",
    "mla_only":         "MLA only",
}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Figure 1: Perplexity bar chart ───────────────────────────────────────────
def fig_perplexity():
    variants = ["dense_ffn", "full", "ssm_only", "transformer_dense", "mla_only"]
    ppl_512  = [154.33, 156.96, 241.31, 269.82, 318.82]
    ppl_1024 = [160.32, 163.73, 250.97, 343.92, 359.53]

    x = np.arange(len(variants))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars1 = ax.bar(x - width/2, ppl_512,  width, label="512-token context",
                   color=[COLORS[v] for v in variants], alpha=0.92)
    bars2 = ax.bar(x + width/2, ppl_1024, width, label="1 024-token context",
                   color=[COLORS[v] for v in variants], alpha=0.50,
                   edgecolor=[COLORS[v] for v in variants], linewidth=1.4)

    # Annotate best bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=15, ha="right")
    ax.set_ylabel("Validation Perplexity  (lower = better)")
    ax.set_title("Language Modelling Quality — All Variants at 0.5B Parameters")
    ax.legend(loc="upper left")

    # Highlight winner
    ax.annotate("Best model", xy=(x[0] - width/2, ppl_512[0]),
                xytext=(x[0] + 0.55, ppl_512[0] + 55),
                arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.5),
                color="#2196F3", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 420)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_perplexity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 2: TTFT scaling lines ─────────────────────────────────────────────
def fig_ttft_scaling():
    contexts = [512, 1024, 2048]
    data = {
        "dense_ffn":        [2560.1,  4507.8,  8390.4],
        "full":             [3363.6,  5220.6,  9049.5],
        "ssm_only":         [3311.8,  5100.4,  8643.1],
        "mla_only":         [1598.5,  1739.2,  2109.5],
        "transformer_dense":[113.9,    256.1,   630.7],
    }

    fig, ax = plt.subplots(figsize=(7, 4.4))

    for key, vals in data.items():
        lw = 2.6 if key in ("dense_ffn", "mla_only") else 1.5
        ls = "-" if key in ("dense_ffn", "mla_only") else "--"
        ax.plot(contexts, vals, color=COLORS[key], linewidth=lw,
                linestyle=ls, marker="o", markersize=6, label=LABELS[key])

    # Shade the "near-linear" region for mla_only
    mla_vals = data["mla_only"]
    ax.fill_between(contexts, mla_vals, alpha=0.08, color=COLORS["mla_only"])
    ax.text(1900, mla_vals[1] - 400, "Near-linear\nscaling", color=COLORS["mla_only"],
            fontsize=8.5, ha="center", style="italic")

    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Time to first token — TTFT (ms)")
    ax.set_title("Inference Latency on M3 MacBook\nvs. Context Length")
    ax.set_xticks(contexts)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 10500)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_ttft_scaling.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 3: Memory vs Perplexity scatter (Pareto) ──────────────────────────
def fig_pareto():
    pts = {
        "dense_ffn":        (24.88, 154.33),
        "full":             (28.75, 156.96),
        "ssm_only":         (26.50, 241.31),
        "mla_only":         (16.20, 318.82),
        "transformer_dense":(5.36,  269.82),
    }

    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    for key, (mem, ppl) in pts.items():
        ax.scatter(mem, ppl, color=COLORS[key], s=140, zorder=3)
        offset = {"dense_ffn": (-4.5, 8), "full": (0.4, 6),
                  "ssm_only": (0.4, 6), "mla_only": (-5.5, 7),
                  "transformer_dense": (0.4, 6)}[key]
        ax.text(mem + offset[0], ppl + offset[1], LABELS[key],
                color=COLORS[key], fontsize=9, fontweight="bold")

    # Draw Pareto frontier annotation
    ax.annotate("",
                xy=(24.88, 154.33), xytext=(5.36, 269.82),
                arrowprops=dict(arrowstyle="-", color="#888", lw=1,
                                linestyle="dashed"))

    ax.set_xlabel("Peak GPU memory during training (GB)\n(lower = more efficient)")
    ax.set_ylabel("Validation perplexity at 512 tokens\n(lower = better quality)")
    ax.set_title("Quality vs. Memory Trade-off at 0.5B Parameters")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Arrow pointing to winner
    ax.annotate("Pareto-optimal:\nbest quality +\nlowest hybrid memory",
                xy=(24.88, 154.33),
                xytext=(18, 195),
                arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.5),
                color="#2196F3", fontsize=8.5, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2196F3", lw=1))

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_pareto.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 4: Training loss curves ───────────────────────────────────────────
def fig_training_loss():
    steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
    full_val  = [6.572, 6.117, 5.634, 5.480, 5.162, 5.106, 5.093]
    dense_val = [6.544, 6.074, 5.594, 5.443, 5.273, 5.093, 5.074]

    fig, ax = plt.subplots(figsize=(7, 4.2))

    ax.plot(steps, full_val,  color=COLORS["full"],     linewidth=2.2,
            marker="o", markersize=5, label=LABELS["full"])
    ax.plot(steps, dense_val, color=COLORS["dense_ffn"], linewidth=2.2,
            marker="s", markersize=5, label=LABELS["dense_ffn"])

    # Mark curriculum transitions
    for step, ctx, ya in [(3000, "→512 tokens", 5.75), (5000, "→1 024 tokens", 5.75)]:
        ax.axvline(step, color="#999", linestyle=":", linewidth=1.2)
        ax.text(step + 60, ya, ctx, fontsize=8, color="#555", rotation=90, va="top")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss  (lower = better)")
    ax.set_title("Training Progress: TALH Full (MoE) vs. Dense-FFN")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(800, 7200)
    ax.set_ylim(4.9, 6.8)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_training_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 5: Phase-1 compute-matched bar chart (all 5 variants, step 4K) ───
def fig_phase1_matched():
    """
    This is the ONLY fair comparison: all five variants trained for the same
    4,000 steps with the same 256→512 context curriculum.
    PPL = exp(val_loss) at step 4,000.
    """
    variants = ["dense_ffn", "full", "ssm_only", "transformer_dense", "mla_only"]
    ppl = [231.1, 239.8, 238.9, 263.0, 315.1]   # exp of val_loss @ 4K

    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))

    bars = ax.bar(x, ppl, color=[COLORS[v] for v in variants], width=0.55,
                  edgecolor="white", linewidth=0.8, alpha=0.92)

    for bar, p in zip(bars, ppl):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{p:.0f}", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=15, ha="right")
    ax.set_ylabel("Validation Perplexity  (lower = better)")
    ax.set_title("Compute-Matched Quality Comparison\n"
                 "All 5 variants — same steps, same data, same curriculum")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 370)

    # Annotate the gap
    ax.annotate("", xy=(x[0], ppl[0]), xytext=(x[3], ppl[3]),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.5))
    mid_x = (x[0] + x[3]) / 2
    mid_y = (ppl[0] + ppl[3]) / 2 + 12
    ax.text(mid_x, mid_y, "12% gap", ha="center", fontsize=9,
            color="#444", fontweight="bold")

    # Colour legend patches
    patches = [mpatches.Patch(color=COLORS[v], label=LABELS[v]) for v in variants]
    ax.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_phase1_matched.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 6: All-variant training loss curves through Phase-1 ───────────────
def fig_all_variants_phase1():
    """Loss curves for all 5 variants through Phase-1 (steps 1K-4K)."""
    steps = [1000, 2000, 3000, 4000]
    data = {
        "dense_ffn":        [6.544, 6.074, 5.594, 5.443],
        "full":             [6.572, 6.117, 5.634, 5.480],
        "ssm_only":         [6.520, 6.102, 5.618, 5.476],
        "transformer_dense":[6.626, 6.145, 5.790, 5.572],
        "mla_only":         [6.706, 6.336, 5.935, 5.753],
    }

    fig, ax = plt.subplots(figsize=(7, 4.2))

    for key, vals in data.items():
        lw = 2.5 if key in ("dense_ffn", "transformer_dense") else 1.5
        ls = "-" if key in ("dense_ffn", "transformer_dense") else "--"
        ax.plot(steps, vals, color=COLORS[key], linewidth=lw,
                linestyle=ls, marker="o", markersize=5, label=LABELS[key])

    ax.axvline(3000, color="#999", linestyle=":", linewidth=1.2)
    ax.text(3060, 6.55, "→ 512 tokens", fontsize=8, color="#555",
            rotation=90, va="top")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss  (lower = better)")
    ax.set_title("Phase-1 Training: All Variants (Equal Compute Budget)")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(800, 4200)
    ax.set_ylim(5.3, 6.85)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_all_variants_phase1.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Update fig_pareto to use Phase-1 numbers ────────────────────────────────
def fig_pareto_matched():
    """Pareto using compute-matched Phase-1 PPL (step 4K)."""
    pts = {
        "dense_ffn":        (24.01, 231.1),
        "full":             (27.88, 239.8),
        "ssm_only":         (26.50, 238.9),
        "mla_only":         (16.20, 315.1),
        "transformer_dense":(5.36,  263.0),
    }

    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    for key, (mem, ppl) in pts.items():
        ax.scatter(mem, ppl, color=COLORS[key], s=140, zorder=3)
        offset = {
            "dense_ffn":        (-5.0, 6),
            "full":             ( 0.4, 6),
            "ssm_only":         ( 0.4, -14),
            "mla_only":         (-5.5, 7),
            "transformer_dense":( 0.4, 6),
        }[key]
        ax.text(mem + offset[0], ppl + offset[1], LABELS[key],
                color=COLORS[key], fontsize=9, fontweight="bold")

    ax.set_xlabel("Peak GPU memory during training (GB)")
    ax.set_ylabel("Validation perplexity at 512 tokens\n(compute-matched, step 4K — lower = better)")
    ax.set_title("Quality vs. Memory — Compute-Matched Comparison")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax.annotate("Best quality +\nlowest hybrid memory",
                xy=(24.01, 231.1), xytext=(16, 210),
                arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.5),
                color="#2196F3", fontsize=8.5, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2196F3", lw=1))

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_pareto_matched.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    fig_perplexity()
    fig_ttft_scaling()
    fig_pareto()
    fig_training_loss()
    fig_phase1_matched()
    fig_all_variants_phase1()
    fig_pareto_matched()
    print("All figures generated.")
