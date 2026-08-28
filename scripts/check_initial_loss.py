#!/usr/bin/env python3
"""
Quick initial-loss sanity check for TALH before burning GPU budget.

Example:
    python3 scripts/check_initial_loss.py \
        --vocab_size 50257 \
        --d_model 800 --n_layers 12 --num_heads 16 \
        --latent_dim 200 --state_dim 32 --d_ff 1600 \
        --n_experts 8 --top_k 2 --ablation full --device cuda
"""

from __future__ import annotations

import argparse
import math

import torch

from talh.train_torch import TALHTorch, TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check TALH initial loss/logit scale")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--max_expected_ce", type=float, default=12.0)

    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--d_model", type=int, default=800)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--latent_dim", type=int, default=200)
    p.add_argument("--state_dim", type=int, default=32)
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=1600)
    p.add_argument(
        "--ablation",
        default="full",
        choices=["full", "no_ttt", "mla_only", "ssm_only", "dense_ffn", "baseline"],
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    cfg = TrainConfig(
        vocab_size=args.vocab_size,
        n_layers=args.n_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
        state_dim=args.state_dim,
        n_experts=args.n_experts,
        top_k=args.top_k,
        d_ff=args.d_ff,
        ablation=args.ablation,
    )

    model = TALHTorch(cfg).to(device)
    model.eval()

    ce_losses: list[float] = []
    logits_mean_abs: list[float] = []
    logits_max_abs: list[float] = []

    with torch.no_grad():
        for _ in range(args.trials):
            input_ids = torch.randint(
                0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device
            )
            out = model(input_ids, targets=input_ids)
            ce_losses.append(float(out["ce_loss"]))
            logits = out["logits"]
            logits_mean_abs.append(float(logits.abs().mean()))
            logits_max_abs.append(float(logits.abs().max()))

    avg_ce = sum(ce_losses) / len(ce_losses)
    avg_mean_abs = sum(logits_mean_abs) / len(logits_mean_abs)
    avg_max_abs = sum(logits_max_abs) / len(logits_max_abs)
    expected = math.log(cfg.vocab_size)

    print(f"device={device}")
    print(f"expected_ce~={expected:.4f}")
    print(f"avg_ce={avg_ce:.4f}")
    print(f"avg_logits_mean_abs={avg_mean_abs:.4f}")
    print(f"avg_logits_max_abs={avg_max_abs:.4f}")

    if avg_ce > args.max_expected_ce:
        print("FAIL: initial CE loss is too high")
        return 1

    print("PASS: initial CE loss is within the expected range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
