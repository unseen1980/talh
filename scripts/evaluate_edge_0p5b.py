"""
Medium-scale evaluation harness for the TALH 0.5B edge paper package.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from talh.convert_checkpoint import load_for_inference
from talh.train_torch import TALHTorch, TrainConfig, TokenisedDataset, _evaluate

from edge_0p5b_manifest import register_artifact
import eval_ruler


DEFAULT_VARIANTS = ["full", "dense_ffn", "mla_only", "ssm_only", "transformer_dense"]


def resolve_checkpoint(
    checkpoints_root: Path,
    variant: str,
    explicit_path: str | None = None,
) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    base = checkpoints_root / f"talh_{variant}"
    if not base.exists():
        raise FileNotFoundError(f"Variant directory not found: {base}")

    for candidate in ("best.pt", "best.safetensors"):
        path = base / candidate
        if path.exists():
            return path

    latest_pt = sorted(base.glob("step_*.pt"))
    if latest_pt:
        return latest_pt[-1]

    latest_st = sorted(base.glob("step_*.safetensors"))
    if latest_st:
        return latest_st[-1]

    raise FileNotFoundError(f"No checkpoint found for variant '{variant}' in {base}")


def detect_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate_validation(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "edge_0p5b_validation.csv"
    device = detect_device(args.device)

    rows: list[dict] = []
    for variant in args.variants:
        ckpt_path = resolve_checkpoint(Path(args.checkpoints_root), variant)
        model, _ = load_for_inference(ckpt_path, backend="torch")
        if not isinstance(model, TALHTorch):
            raise TypeError(f"Expected PyTorch TALHTorch for validation, got {type(model)}")
        model = model.to(device)
        model.eval()

        for seq_len in args.seq_lens:
            dataset = TokenisedDataset(
                args.dataset,
                split="validation",
                seq_len=seq_len,
                tokenizer_name=args.tokenizer_name,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=device.startswith("cuda"),
            )
            val_loss = _evaluate(model, loader, device, n_batches=args.val_batches)
            ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            rows.append({
                "variant": variant,
                "checkpoint": str(ckpt_path),
                "device": device,
                "dataset": args.dataset,
                "tokenizer": args.tokenizer_name,
                "seq_len": seq_len,
                "val_batches": args.val_batches,
                "val_loss": round(val_loss, 6),
                "perplexity": round(ppl, 6) if math.isfinite(ppl) else "inf",
            })
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    register_artifact(
        args.manifest_path,
        artifact_path=out_csv,
        artifact_type="csv",
        source_script="scripts/evaluate_edge_0p5b.py validate",
        stage="stage2_medium_validation",
        metadata={
            "variants": args.variants,
            "seq_lens": args.seq_lens,
            "device": device,
            "dataset": args.dataset,
            "tokenizer": args.tokenizer_name,
            "checkpoints_root": args.checkpoints_root,
        },
    )
    return out_csv


def evaluate_retrieval(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "edge_0p5b_retrieval.csv"
    tokenizer = eval_ruler.load_tokenizer(args.tokenizer_name)
    device = detect_device(args.device)

    rows: list[dict] = []
    for variant in args.variants:
        ckpt_path = resolve_checkpoint(Path(args.checkpoints_root), variant)
        model, backend = load_for_inference(ckpt_path, backend=args.backend)
        if backend == "torch":
            model = model.to(device)
            model.eval()

        for context_len in args.context_lens:
            result = eval_ruler.evaluate_niah(
                model,
                backend=backend,
                tokenizer=tokenizer,
                context_len=context_len,
                n_samples=args.n_samples,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
            )
            rows.append({
                "variant": variant,
                "checkpoint": str(ckpt_path),
                "backend": backend,
                "device": device if backend == "torch" else backend,
                "tokenizer": args.tokenizer_name,
                **result,
            })

        if backend == "torch" and device.startswith("cuda"):
            torch.cuda.empty_cache()

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    register_artifact(
        args.manifest_path,
        artifact_path=out_csv,
        artifact_type="csv",
        source_script="scripts/evaluate_edge_0p5b.py retrieval",
        stage="stage2_medium_retrieval",
        metadata={
            "variants": args.variants,
            "context_lens": args.context_lens,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "backend": args.backend,
            "tokenizer": args.tokenizer_name,
            "checkpoints_root": args.checkpoints_root,
        },
    )
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge 0.5B evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoints_root", default="experiments/medium_budget/checkpoints")
    common.add_argument("--output_dir", default="experiments/medium/results")
    common.add_argument("--manifest_path", default="experiments/medium/results/edge_0p5b_manifest.json")
    common.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    common.add_argument("--tokenizer_name", default="gpt2")

    val = sub.add_parser("validate", parents=[common], help="Compute validation loss/perplexity")
    val.add_argument("--dataset", default="fineweb")
    val.add_argument("--seq_lens", nargs="+", type=int, default=[512, 1024])
    val.add_argument("--val_batches", type=int, default=25)
    val.add_argument("--batch_size", type=int, default=4)
    val.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    ret = sub.add_parser("retrieval", parents=[common], help="Run tokenizer-backed retrieval eval")
    ret.add_argument("--context_lens", nargs="+", type=int, default=[512, 1024, 2048])
    ret.add_argument("--n_samples", type=int, default=50)
    ret.add_argument("--seed", type=int, default=42)
    ret.add_argument("--max_new_tokens", type=int, default=8)
    ret.add_argument("--backend", default="auto", choices=["auto", "torch", "mlx"])
    ret.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        out_csv = evaluate_validation(args)
    elif args.command == "retrieval":
        out_csv = evaluate_retrieval(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
