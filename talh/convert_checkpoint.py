"""
TALH Checkpoint Conversion Utilities.

Converts between:
  - PyTorch .pt checkpoints  (saved by talh/train_torch.py)
  - MLX .safetensors          (saved by talh/train_mlx.py, loaded by talh/model.py)

The two formats use **identical parameter key naming** (dot-notation hierarchy,
e.g. ``layers.0.mla.q_proj.weight``). Conversion only requires changing the
tensor representation — no key remapping is needed.

Functions
---------
pt_to_safetensors   Convert a .pt checkpoint to MLX-compatible .safetensors.
safetensors_to_pt   Convert a .safetensors checkpoint to PyTorch .pt.
load_for_inference  Load a checkpoint into the correct model class (auto-detect
                    backend based on platform).

Example
-------
    # On Colab after training:
    from talh.convert_checkpoint import pt_to_safetensors
    pt_to_safetensors("checkpoints/talh_full/best.pt",
                      "checkpoints/talh_full/best.safetensors")

    # On Mac / any platform:
    from talh.convert_checkpoint import load_for_inference
    model, backend = load_for_inference("checkpoints/talh_full/best.pt")
    print(f"Loaded on: {backend}")
"""

import json
import logging
import platform
import re
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key remapping: PyTorch state_dict → MLX parameter paths
# ---------------------------------------------------------------------------

# In PyTorch (TALHLayerTorch), MinimalSSMTorch is stored directly as self.ssm:
#   layers.i.ssm.in_proj.weight
#
# In MLX (TALHLayer), SSMBranch stores MinimalSSM as self.core:
#   layers.i.ssm.core.in_proj.weight
#
# All other component paths are identical between the two implementations.
_SSM_DIRECT = re.compile(r"^(layers\.\d+\.ssm)\.(?!core\.)")


def _remap_pt_to_mlx(key: str) -> str:
    """Remap a PyTorch state_dict key to the equivalent MLX parameter path."""
    return _SSM_DIRECT.sub(r"\1.core.", key)


def _remap_mlx_to_pt(key: str) -> str:
    """Reverse: remap an MLX parameter path to a PyTorch state_dict key."""
    return key.replace(".ssm.core.", ".ssm.")


# Fields that live in TrainConfig but NOT in TALHConfig.
# These must be stripped before constructing a TALHConfig.
_TRAIN_ONLY_KEYS = frozenset({
    "dataset", "seq_len", "max_seq_len", "batch_size", "max_steps",
    "eval_every", "save_every", "lr", "weight_decay", "grad_clip",
    "warmup_steps", "aux_loss_coef", "curriculum", "curriculum_steps",
    "output_dir", "seed", "ablation",
})


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------

def pt_to_safetensors(pt_path: str | Path, out_path: str | Path) -> None:
    """
    Convert a PyTorch .pt checkpoint to an MLX-compatible .safetensors file.

    The checkpoint produced by ``talh/train_torch.py`` has the structure::

        {"step": int, "model": OrderedDict(...), "config": dict}

    Each ``model`` key maps directly to the equivalent MLX parameter path,
    so no remapping is required — only the tensor format changes.

    Args:
        pt_path:  Path to the source .pt checkpoint.
        out_path: Destination .safetensors path (parent dirs created if needed).

    Raises:
        FileNotFoundError: If ``pt_path`` does not exist.
        ImportError:       If ``mlx`` is not installed.
        KeyError:          If the checkpoint has no ``"model"`` key.
    """
    import torch

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for pt_to_safetensors. "
            "Install it with:  python -m pip install safetensors"
        ) from exc

    pt_path  = Path(pt_path)
    out_path = Path(out_path)

    if not pt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

    logger.info("Loading PyTorch checkpoint: %s", pt_path)
    # weights_only=False because the checkpoint may contain non-tensor objects
    # (config dict, step int). map_location="cpu" avoids CUDA dependency.
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    if "model" not in ckpt:
        raise KeyError(
            f"Expected checkpoint to have a 'model' key. "
            f"Found: {list(ckpt.keys())}"
        )

    state = ckpt["model"]
    flat: dict[str, "torch.Tensor"] = {}
    seen_data_ptrs: set[int] = set()
    for key, tensor in state.items():
        mlx_key = _remap_pt_to_mlx(key)
        t = tensor.float().contiguous()
        # Clone tensors that share storage (e.g. tied embed/lm_head weights)
        # so safetensors doesn't reject duplicate memory regions.
        ptr = t.data_ptr()
        if ptr in seen_data_ptrs:
            t = t.clone()
        seen_data_ptrs.add(ptr)
        flat[mlx_key] = t

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(flat, str(out_path))
    logger.info("Saved %d tensors → %s", len(flat), out_path)


def safetensors_to_pt(st_path: str | Path, out_path: str | Path) -> None:
    """
    Convert an MLX .safetensors checkpoint to a PyTorch .pt file.

    Useful for loading MLX-trained weights into the PyTorch model for
    CPU inference or Colab-based evaluation.

    Args:
        st_path:  Path to the source .safetensors file.
        out_path: Destination .pt path.

    Raises:
        FileNotFoundError: If ``st_path`` does not exist.
        ImportError:       If ``safetensors`` is not installed.
    """
    import torch

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "safetensors is required. "
            "Install it with:  python -m pip install safetensors"
        ) from exc

    st_path  = Path(st_path)
    out_path = Path(out_path)

    if not st_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {st_path}")

    state: dict[str, "torch.Tensor"] = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            pt_key = _remap_mlx_to_pt(key)
            state[pt_key] = f.get_tensor(key)

    # Recover config from sibling config.json if present
    config_path = st_path.parent / "config.json"
    config: dict = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 0, "model": state, "config": config}, str(out_path))
    logger.info("Saved %d tensors → %s", len(state), out_path)


# ---------------------------------------------------------------------------
# Inference loader
# ---------------------------------------------------------------------------

def _detect_backend() -> Literal["mlx", "torch"]:
    """Return the preferred inference backend for the current platform."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401 — only checking availability
            return "mlx"
        except ImportError:
            pass
    return "torch"


def load_for_inference(
    checkpoint_path: str | Path,
    config: dict | None = None,
    backend: Literal["mlx", "torch", "auto"] = "auto",
) -> tuple:
    """
    Load a TALH checkpoint for inference, selecting the right backend.

    Accepts either ``.pt`` (PyTorch) or ``.safetensors`` (MLX) checkpoints.
    When the required format for the chosen backend is missing, it is
    automatically generated via the format converters above.

    Args:
        checkpoint_path: Path to a ``.pt`` or ``.safetensors`` checkpoint.
        config:          Optional model-config dict. When ``None`` the config
                         is read from a ``config.json`` sibling file, or from
                         the ``"config"`` key inside a ``.pt`` checkpoint.
        backend:         ``"auto"`` → MLX on Apple Silicon, PyTorch elsewhere.
                         ``"mlx"``  → Force MLX (raises if unavailable).
                         ``"torch"``→ Force PyTorch.

    Returns:
        ``(model, backend_str)`` where ``backend_str`` is ``"mlx"`` or
        ``"torch"``.
    """
    checkpoint_path = Path(checkpoint_path)
    if backend == "auto":
        backend = _detect_backend()

    # The MLX model only supports the "full" architecture (no ablation
    # variants like IdentityBranch).  Detect ablation from the checkpoint
    # config and fall back to PyTorch for non-full variants.
    if backend == "mlx":
        raw_cfg = _resolve_config(checkpoint_path, config)
        ablation = raw_cfg.get("ablation", "full")
        if ablation != "full":
            logger.info(
                "Ablation '%s' is not supported by the MLX model — "
                "falling back to PyTorch backend.",
                ablation,
            )
            backend = "torch"

    if backend == "mlx":
        model = _load_mlx(checkpoint_path, config)
        return model, "mlx"
    else:
        model = _load_torch(checkpoint_path, config)
        return model, "torch"


def _resolve_config(
    checkpoint_path: Path,
    config_override: dict | None,
) -> dict:
    """Read model config from override, embedded .pt config, or sibling JSON.

    For .pt checkpoints the embedded config is preferred over config.json
    because it is guaranteed to match the saved weights (config.json may
    have been overwritten by a different training run or backend).
    """
    if config_override is not None:
        return config_override

    # .pt checkpoints embed the config used during training — always
    # prefer this over config.json (which may be stale / overwritten).
    # Check both the given path and a sibling .pt if given .safetensors.
    pt_path = checkpoint_path if checkpoint_path.suffix == ".pt" else (
        checkpoint_path.with_suffix(".pt")
    )
    if pt_path.exists():
        import torch
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        embedded = ckpt.get("config", {})
        del ckpt
        if embedded:
            return embedded

    config_json = checkpoint_path.parent / "config.json"
    if config_json.exists():
        return json.loads(config_json.read_text())

    logger.warning("No config source found — using default TALHConfig")
    return {}


def _load_mlx(checkpoint_path: Path, config_override: dict | None):
    """Load checkpoint into an MLX TALH model."""
    try:
        import mlx.core as mx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MLX is not installed. Use backend='torch' for CPU inference."
        ) from exc

    from talh.model import TALH, TALHConfig

    # Resolve config BEFORE converting, so .pt embedded config is available
    original_path = checkpoint_path

    # Ensure we have a .safetensors file
    if checkpoint_path.suffix == ".pt":
        st_path = checkpoint_path.with_suffix(".safetensors")
        if not st_path.exists():
            logger.info("Auto-converting %s → .safetensors …", checkpoint_path.name)
            pt_to_safetensors(checkpoint_path, st_path)
        checkpoint_path = st_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_cfg = _resolve_config(original_path, config_override)
    # Keep only keys that TALHConfig accepts
    model_kw = {
        k: v for k, v in raw_cfg.items()
        if k in TALHConfig.__dataclass_fields__ and k not in _TRAIN_ONLY_KEYS
    }
    cfg = TALHConfig(**model_kw)
    model = TALH(cfg)

    import mlx.core as mx

    # Load weight dict and apply key remapping.
    # This handles both:
    #   - Freshly converted files (already have .core. keys) — remap is a no-op
    #   - Legacy .safetensors files with raw PyTorch keys (layers.i.ssm.*)
    raw_weights = mx.load(str(checkpoint_path))
    weights = {_remap_pt_to_mlx(k): v for k, v in raw_weights.items()}
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    logger.info("MLX model loaded from %s", checkpoint_path)
    return model


def _load_torch(checkpoint_path: Path, config_override: dict | None):
    """Load checkpoint into a PyTorch TALHTorch model."""
    import torch

    from talh.train_torch import TALHTorch, TrainConfig

    # Ensure we have a .pt file
    if checkpoint_path.suffix == ".safetensors":
        pt_path = checkpoint_path.with_suffix(".pt")
        if not pt_path.exists():
            logger.info("Auto-converting %s → .pt …", checkpoint_path.name)
            safetensors_to_pt(checkpoint_path, pt_path)
        checkpoint_path = pt_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    raw_cfg = config_override or ckpt.get("config", {})

    cfg_kw = {
        k: v for k, v in raw_cfg.items()
        if k in TrainConfig.__dataclass_fields__
    }
    cfg = TrainConfig(**cfg_kw)
    model = TALHTorch(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    logger.info("PyTorch model loaded from %s", checkpoint_path)
    return model
