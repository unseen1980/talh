"""
Checkpoint-backed local latency and memory benchmark for the 0.5B edge paper.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edge_0p5b_manifest import register_artifact
from talh.convert_checkpoint import load_for_inference


DEFAULT_VARIANTS = ["full", "dense_ffn", "transformer_dense"]
PROMPT_TEXT = (
    "The following notes describe a technical design review for a local language model. "
    "Each section discusses latency, memory, and retrieval tradeoffs on personal hardware. "
)


def load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10 ** 9
    return tokenizer


def resolve_checkpoint(checkpoints_root: Path, variant: str) -> Path:
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


def build_prompt_ids(tokenizer, prompt_len: int) -> list[int]:
    base_ids = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    repeats = prompt_len // max(1, len(base_ids)) + 2
    prompt_ids = (base_ids * repeats)[:prompt_len]
    return prompt_ids


def current_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return rss / 1e6
        return rss / 1024.0


def mlx_peak_memory_mb() -> float:
    import mlx.core as mx

    try:
        return mx.get_peak_memory() / 1e6
    except AttributeError:
        try:
            return mx.metal.get_peak_memory() / 1e6
        except AttributeError:
            return 0.0


def reset_mlx_peak_memory() -> None:
    import mlx.core as mx

    try:
        mx.reset_peak_memory()
    except AttributeError:
        try:
            mx.metal.reset_peak_memory()
        except AttributeError:
            pass


def measure_mlx(model, prompt_ids: list[int], n_tokens: int) -> tuple[float, float, float]:
    import mlx.core as mx

    x = mx.array([prompt_ids], dtype=mx.uint32)
    reset_mlx_peak_memory()
    t0 = time.perf_counter()
    logits, kv, ssm_states, _ = model(x)
    mx.eval(logits)
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    next_id = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_id)

    latencies = []
    for _ in range(n_tokens):
        t1 = time.perf_counter()
        logits, kv, ssm_states, _ = model(next_id, kv_caches=kv, ssm_states=ssm_states)
        mx.eval(logits)
        latencies.append((time.perf_counter() - t1) * 1000.0)
        next_id = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_id)

    peak_mb = mlx_peak_memory_mb()
    tpot_ms = sum(latencies) / max(1, len(latencies))
    return ttft_ms, tpot_ms, peak_mb


def measure_torch(model, prompt_ids: list[int], n_tokens: int, device: str) -> tuple[float, float, float]:
    import torch

    model = model.to(device)
    model.eval()
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    before_mb = current_rss_mb()
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(x)
        logits = out["logits"]
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        after_prefill_mb = current_rss_mb()

        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        latencies = []
        for _ in range(n_tokens):
            t1 = time.perf_counter()
            x = torch.cat([x, next_id], dim=1)
            out = model(x)
            logits = out["logits"]
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            latencies.append((time.perf_counter() - t1) * 1000.0)

    peak_mb = max(before_mb, after_prefill_mb, current_rss_mb())
    tpot_ms = sum(latencies) / max(1, len(latencies))
    return ttft_ms, tpot_ms, peak_mb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint-backed local benchmark")
    parser.add_argument("--checkpoints_root", default="experiments/medium_budget/checkpoints")
    parser.add_argument("--output_dir", default="experiments/medium/results")
    parser.add_argument("--manifest_path", default="experiments/medium/results/edge_0p5b_manifest.json")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--prompt_lens", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--n_tokens", type=int, default=32)
    parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"])
    parser.add_argument("--tokenizer_name", default="gpt2")
    parser.add_argument("--torch_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--device_label", default=f"{platform.system()}-{platform.machine()}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latency_csv = output_dir / "edge_0p5b_local_latency.csv"
    memory_csv = output_dir / "edge_0p5b_local_memory.csv"
    tokenizer = load_tokenizer(args.tokenizer_name)

    latency_rows: list[dict] = []
    memory_rows: list[dict] = []

    for variant in args.variants:
        checkpoint = resolve_checkpoint(Path(args.checkpoints_root), variant)
        model, backend = load_for_inference(checkpoint, backend=args.backend)
        variant_rows: list[dict] = []

        for prompt_len in args.prompt_lens:
            prompt_ids = build_prompt_ids(tokenizer, prompt_len)
            status = "ok"
            error = ""
            ttft_ms = tpot_ms = peak_mb = 0.0

            try:
                if backend == "mlx":
                    ttft_ms, tpot_ms, peak_mb = measure_mlx(model, prompt_ids, args.n_tokens)
                    device_used = "mlx"
                else:
                    ttft_ms, tpot_ms, peak_mb = measure_torch(model, prompt_ids, args.n_tokens, args.torch_device)
                    device_used = args.torch_device
            except Exception as exc:
                status = "error"
                error = str(exc)
                device_used = backend

            toks_per_sec = 1000.0 / tpot_ms if status == "ok" and tpot_ms > 0 else 0.0
            latency_row = {
                "variant": variant,
                "checkpoint": str(checkpoint),
                "backend": backend,
                "device": device_used,
                "device_label": args.device_label,
                "tokenizer": args.tokenizer_name,
                "prompt_len": prompt_len,
                "n_decode_tokens": args.n_tokens,
                "ttft_ms": round(ttft_ms, 4),
                "tpot_ms": round(tpot_ms, 4),
                "tokens_per_sec": round(toks_per_sec, 4),
                "status": status,
                "error": error,
            }
            memory_row = {
                "variant": variant,
                "checkpoint": str(checkpoint),
                "backend": backend,
                "device": device_used,
                "device_label": args.device_label,
                "tokenizer": args.tokenizer_name,
                "prompt_len": prompt_len,
                "peak_memory_mb": round(peak_mb, 4),
                "status": status,
                "error": error,
            }
            latency_rows.append(latency_row)
            memory_rows.append(memory_row)
            variant_rows.append(memory_row)

        max_prompt = max(
            (row["prompt_len"] for row in variant_rows if row["status"] == "ok"),
            default=0,
        )
        for row in memory_rows:
            if row["variant"] == variant:
                row["max_usable_prompt_len"] = max_prompt

        del model
        gc.collect()

    with open(latency_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=latency_rows[0].keys())
        writer.writeheader()
        writer.writerows(latency_rows)

    with open(memory_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=memory_rows[0].keys())
        writer.writeheader()
        writer.writerows(memory_rows)

    register_artifact(
        args.manifest_path,
        artifact_path=latency_csv,
        artifact_type="csv",
        source_script="scripts/benchmark_local_inference.py",
        stage="stage3_local_latency",
        metadata={
            "variants": args.variants,
            "prompt_lens": args.prompt_lens,
            "n_tokens": args.n_tokens,
            "backend": args.backend,
            "device_label": args.device_label,
            "tokenizer": args.tokenizer_name,
        },
    )
    register_artifact(
        args.manifest_path,
        artifact_path=memory_csv,
        artifact_type="csv",
        source_script="scripts/benchmark_local_inference.py",
        stage="stage3_local_memory",
        metadata={
            "variants": args.variants,
            "prompt_lens": args.prompt_lens,
            "backend": args.backend,
            "device_label": args.device_label,
            "tokenizer": args.tokenizer_name,
        },
    )

    print(f"Saved: {latency_csv}")
    print(f"Saved: {memory_csv}")


if __name__ == "__main__":
    main()
