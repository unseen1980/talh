"""
H3 / H4 - Tokenizer-backed long-context retrieval evaluation.

Runs a Needle-in-a-Haystack style exact-recall task using real GPT-2 token
ids instead of synthetic character ids. This keeps the evaluation aligned
with the actual training tokenizer and makes the outputs usable in a paper.

Supports both MLX (Apple Silicon) and PyTorch (GPU/CPU) backends via
``load_for_inference`` from ``talh.convert_checkpoint``.
"""

import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# NIAH task generator
# ---------------------------------------------------------------------------

HAYSTACK_SENTENCE = (
    "The meeting was scheduled for Thursday and the project remained on track. "
    "Engineers reviewed the design, documented the changes, and closed the open issues. "
    "The team prepared status updates, confirmed the deadlines, and shared the notes. "
)
NEEDLE_TEMPLATE = "The secret number is {value}. "
QUESTION_TEMPLATE = "Question: What is the secret number? Answer with just the number."


def make_niah_sample(
    context_len: int,
    tokenizer,
    needle_position: float = 0.5,
    rng: random.Random | None = None,
) -> dict:
    """
    Create a Needle-in-a-Haystack sample using the real tokenizer.

    Args:
        context_len:     Total number of tokens in the context.
        needle_position: Where in [0, 1] to insert the needle.
        rng:             Optional seeded random generator.

    Returns:
        dict with keys: 'input_ids', 'needle_value', 'needle_pos_tokens'
    """
    if rng is None:
        rng = random.Random()

    needle_value = rng.randint(1000, 9999)
    haystack_ids = tokenizer.encode(HAYSTACK_SENTENCE, add_special_tokens=False)
    needle_text = NEEDLE_TEMPLATE.format(value=needle_value)
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    question_ids = tokenizer.encode(QUESTION_TEMPLATE, add_special_tokens=False)

    haystack_budget = max(16, context_len - len(needle_ids) - len(question_ids))
    repeats = haystack_budget // max(1, len(haystack_ids)) + 2
    haystack_trimmed = (haystack_ids * repeats)[:haystack_budget]

    needle_pos = min(len(haystack_trimmed), int(needle_position * len(haystack_trimmed)))
    full_context = haystack_trimmed[:needle_pos] + needle_ids + haystack_trimmed[needle_pos:] + question_ids

    return {
        "input_ids": full_context[:context_len],
        "needle_value": needle_value,
        "needle_text": needle_text.strip(),
        "needle_pos_tokens": needle_pos,
    }


# ---------------------------------------------------------------------------
# Retrieval check (greedy decode, look for needle value in output)
# ---------------------------------------------------------------------------

def check_retrieval(output_text: str, needle_value: int) -> bool:
    """
    Check if the needle value appears in the decoded output.
    """
    return str(needle_value) in output_text


# ---------------------------------------------------------------------------
# Backend-specific greedy decode
# ---------------------------------------------------------------------------

def _greedy_decode_mlx(model, input_ids_list: list[int], max_new_tokens: int = 10) -> list[int]:
    """Greedy decode using MLX backend."""
    import mlx.core as mx

    generated = []
    x = mx.array([input_ids_list], dtype=mx.uint32)
    kv_caches  = None
    ssm_states = None

    for _ in range(max_new_tokens):
        logits, kv_caches, ssm_states, _ = model(
            x, kv_caches=kv_caches, ssm_states=ssm_states
        )
        next_id = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_id)
        generated.append(next_id.item())
        x = next_id

    return generated


def _greedy_decode_torch(model, input_ids_list: list[int], max_new_tokens: int = 10) -> list[int]:
    """Greedy decode using PyTorch backend."""
    import torch

    generated = []
    device = next(model.parameters()).device
    x = torch.tensor([input_ids_list], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(x)
            logits = out["logits"]
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_id.item())
            x = torch.cat([x, next_id], dim=1)

    return generated


def load_tokenizer(tokenizer_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for tokenizer-backed retrieval eval. "
            "Install it with: python -m pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10 ** 9
    return tokenizer


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_niah(
    model,
    backend: str,
    tokenizer,
    context_len: int,
    n_samples: int,
    needle_positions: list[float] | None = None,
    seed: int = 42,
    max_new_tokens: int = 8,
) -> dict:
    """
    Run NIAH evaluation at a given context length.

    Args:
        model:       Loaded model (MLX TALH or PyTorch TALHTorch).
        backend:     "mlx" or "torch".
        context_len: Context length to test.
        n_samples:   Number of samples to evaluate.
        needle_positions: Needle insertion positions (fractions 0-1).
        seed:        Random seed for reproducibility.
        max_new_tokens: Number of decode tokens for the answer.

    Returns:
        dict with 'accuracy', 'avg_ttft_ms', 'context_len', 'n_samples'.
    """
    if needle_positions is None:
        needle_positions = [0.1, 0.3, 0.5, 0.7, 0.9]

    decode_fn = _greedy_decode_mlx if backend == "mlx" else _greedy_decode_torch

    rng = random.Random(seed)
    correct = 0
    ttfts   = []

    for i in range(n_samples):
        pos = needle_positions[i % len(needle_positions)]
        sample = make_niah_sample(
            context_len=context_len,
            tokenizer=tokenizer,
            needle_position=pos,
            rng=rng,
        )

        input_ids = sample["input_ids"][:context_len]

        # TTFT
        t0 = time.perf_counter()
        output_ids = decode_fn(model, input_ids, max_new_tokens=max_new_tokens)
        t1 = time.perf_counter()
        ttfts.append((t1 - t0) * 1000)
        output_text = tokenizer.decode(output_ids)

        if check_retrieval(output_text, sample["needle_value"]):
            correct += 1

    return {
        "context_len":  context_len,
        "n_samples":    n_samples,
        "accuracy":     correct / n_samples,
        "avg_ttft_ms":  statistics.mean(ttfts),
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args: argparse.Namespace) -> tuple:
    """
    Load model for evaluation.

    Returns:
        (model, backend_str, vocab_size, ablation_name)
    """
    if args.checkpoint and not args.untrained:
        from talh.convert_checkpoint import load_for_inference

        backend_arg = getattr(args, "backend", "auto")
        model, backend = load_for_inference(
            args.checkpoint, backend=backend_arg,
        )

        # Extract vocab_size from the model's config
        if backend == "mlx":
            vocab_size = model.config.vocab_size
            ablation = "full"
        else:
            vocab_size = model.cfg.vocab_size
            ablation = model.cfg.ablation

        print(f"[INFO] Loaded checkpoint: {args.checkpoint} (backend={backend})")
        return model, backend, vocab_size, ablation

    # --untrained: create a randomly-initialised model for structural testing
    backend = getattr(args, "backend", "auto")
    if backend == "auto":
        from talh.convert_checkpoint import _detect_backend
        backend = _detect_backend()

    if backend == "mlx":
        from talh.model import TALH, TALHConfig
        cfg = TALHConfig(
            n_layers=args.n_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            latent_dim=args.latent_dim,
            state_dim=args.state_dim,
        )
        model = TALH(cfg)
        vocab_size = cfg.vocab_size
        ablation = "full"
    else:
        from talh.train_torch import TALHTorch, TrainConfig
        cfg = TrainConfig(
            n_layers=args.n_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            latent_dim=args.latent_dim,
            state_dim=args.state_dim,
            ablation=args.ablation,
        )
        model = TALHTorch(cfg)
        model.eval()
        vocab_size = cfg.vocab_size
        ablation = cfg.ablation

    print(f"[INFO] Created untrained model (backend={backend})")
    return model, backend, vocab_size, ablation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_name)
    model, backend, vocab_size, ablation = load_model(args)

    print(f"\n=== NIAH Evaluation: {ablation} (backend={backend}) ===")
    print(f"{'Context Len':>12} | {'Accuracy':>10} | {'Avg TTFT (ms)':>14}")
    print("-" * 44)

    rows = []
    for clen in args.context_lens:
        result = evaluate_niah(
            model,
            backend=backend,
            tokenizer=tokenizer,
            context_len=clen,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
        )
        pct = result["accuracy"] * 100
        print(f"{clen:>12,} | {pct:>9.1f}% | {result['avg_ttft_ms']:>14.1f}")
        rows.append({
            "ablation": ablation,
            "backend": backend,
            "checkpoint": args.checkpoint or "",
            "tokenizer": args.tokenizer_name,
            **result,
        })

    csv_path = out_dir / f"niah_{ablation}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H3/H4: RULER / NIAH evaluation")
    p.add_argument("--checkpoint",    default=None,
                   help="Path to .pt or .safetensors checkpoint")
    p.add_argument("--untrained",     action="store_true",
                   help="Use randomly-initialised model (for structural testing)")
    p.add_argument("--backend",       default="auto",
                   choices=["auto", "mlx", "torch"],
                   help="Inference backend (default: auto-detect)")
    p.add_argument("--ablation",      default="full",
                   choices=["full", "mla_only", "ssm_only", "dense_ffn", "baseline", "transformer_dense"])
    p.add_argument("--context_lens",  nargs="+", type=int, default=[512, 1024, 4096])
    p.add_argument("--n_samples",     type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--tokenizer_name", default="gpt2")
    p.add_argument("--n_layers",      type=int, default=4)
    p.add_argument("--d_model",       type=int, default=256)
    p.add_argument("--num_heads",     type=int, default=4)
    p.add_argument("--latent_dim",    type=int, default=64)
    p.add_argument("--state_dim",     type=int, default=16)
    p.add_argument("--output_dir",    default="results/")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
