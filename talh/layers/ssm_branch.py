"""
SSM Branch — wrapper around Mamba-2 / Gated DeltaNet.

On Mac (MLX) this module provides a pure-MLX minimal SSM implementation
for local benchmarking and unit testing. On Colab/GPU with the `mamba-ssm`
or `fla` libraries installed you can swap in the full CUDA kernels via the
`backend` argument.

The key property being tested (H4): the recurrent state size is FIXED
regardless of sequence length, giving O(1) memory growth vs. O(T) for
attention-based KV caches.

References:
    Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient
    Algorithms Through Structured State Space Duality", 2024. (Mamba-2)

    Yang et al., "Gated Delta Networks: Improving Mamba2 with
    Delta Rule", 2024. (Gated DeltaNet)
"""

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# MLX minimal SSM (for local testing / ablation)
# ---------------------------------------------------------------------------

class MinimalSSM(nn.Module):
    """
    Minimal selective state-space model in pure MLX.

    This is a simplified implementation for unit-testing and memory
    benchmarking on Mac. It captures the essential property: the hidden
    state `h` has fixed size (state_dim) regardless of sequence length T.

    It is NOT a faithful Mamba-2 or Gated DeltaNet — use `mamba-ssm` /
    `fla` on Colab for research-quality results.

    Architecture:
        For each time step t:
            h_t = A_t * h_{t-1} + B_t * x_t
            y_t = C_t * h_t

        where A_t, B_t, C_t are input-dependent (selective) projections.

    Args:
        d_model:   Input/output feature dimension.
        state_dim: Recurrent state size (fixed, independent of seq len).
        expand:    Expansion factor for the inner dimension.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 16,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Selective SSM parameter projections (input-dependent A, B, C)
        self.A_proj = nn.Linear(self.d_inner, state_dim, bias=False)
        self.B_proj = nn.Linear(self.d_inner, state_dim, bias=False)
        self.C_proj = nn.Linear(self.d_inner, state_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.RMSNorm(self.d_inner)

    @property
    def state_bytes(self) -> int:
        """Fixed recurrent state size in bytes (float32)."""
        return self.state_dim * self.d_inner * 4

    def __call__(
        self,
        x: mx.array,
        state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass over a sequence.

        Args:
            x:     Input, shape (batch, seq_len, d_model).
            state: Optional initial recurrent state,
                   shape (batch, d_inner, state_dim). Defaults to zeros.

        Returns:
            (output, final_state)
            output:      Shape (batch, seq_len, d_model).
            final_state: Shape (batch, d_inner, state_dim) — fixed size.
        """
        B, T, _ = x.shape

        # Project input to inner dim
        xz = self.in_proj(x)                      # (B, T, d_inner*2)
        x_in, z = xz[..., :self.d_inner], xz[..., self.d_inner:]
        x_in = nn.silu(x_in) * mx.sigmoid(z)      # gated activation

        # Selective SSM parameters
        A = -mx.exp(self.A_proj(x_in))             # (B, T, state_dim) — stable negative
        B_ssm = self.B_proj(x_in)                  # (B, T, state_dim)
        C_ssm = self.C_proj(x_in)                  # (B, T, state_dim)

        # Initialise state
        if state is None:
            state = mx.zeros((B, self.d_inner, self.state_dim))

        # Recurrent scan (sequential for correctness; parallelisable via
        # associative scan on GPU / Colab backend)
        outputs = []
        for t in range(T):
            a_t = mx.exp(A[:, t, :])               # (B, state_dim) — discretised
            b_t = B_ssm[:, t, :]                   # (B, state_dim)
            x_t = x_in[:, t, :]                    # (B, d_inner)
            c_t = C_ssm[:, t, :]                   # (B, state_dim)

            # State update: h = a * h + b * x (outer product for state update)
            # state: (B, d_inner, state_dim)
            state = (
                state * a_t[:, None, :]             # gated by a_t
                + x_t[:, :, None] * b_t[:, None, :] # outer product
            )
            y_t = mx.sum(state * c_t[:, None, :], axis=-1)  # (B, d_inner)
            outputs.append(y_t)

        out = mx.stack(outputs, axis=1)            # (B, T, d_inner)
        out = self.norm(out)
        out = self.out_proj(out)                   # (B, T, d_model)

        return out, state


# ---------------------------------------------------------------------------
# SSMBranch — public interface (dispatches to backend)
# ---------------------------------------------------------------------------

class SSMBranch(nn.Module):
    """
    SSM Branch for the TALH parallel hybrid block.

    Wraps MinimalSSM for Mac/MLX use. On Colab with CUDA libraries available
    set backend='mamba2' or backend='deltanet' to use production kernels.

    Args:
        d_model:   Hidden dimension.
        state_dim: Recurrent state size (fixed).
        backend:   'minimal' (MLX, default) | 'mamba2' | 'deltanet'
    """

    SUPPORTED_BACKENDS = {"minimal", "mamba2", "deltanet"}

    def __init__(
        self,
        d_model: int,
        state_dim: int = 16,
        backend: str = "minimal",
    ) -> None:
        super().__init__()
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {self.SUPPORTED_BACKENDS}, got '{backend}'"
            )
        self.backend = backend
        self.d_model = d_model
        self.state_dim = state_dim

        if backend == "minimal":
            self.core = MinimalSSM(d_model=d_model, state_dim=state_dim)
        elif backend == "mamba2":
            self.core = self._load_mamba2(d_model, state_dim)
        elif backend == "deltanet":
            self.core = self._load_deltanet(d_model, state_dim)

    @staticmethod
    def _load_mamba2(d_model: int, state_dim: int):
        """Load Mamba-2 from mamba-ssm (requires CUDA)."""
        try:
            from mamba_ssm import Mamba2
            return Mamba2(d_model=d_model, d_state=state_dim)
        except ImportError as exc:
            raise ImportError(
                "mamba-ssm is not installed. "
                "Run: pip install mamba-ssm (requires CUDA)."
            ) from exc

    @staticmethod
    def _load_deltanet(d_model: int, state_dim: int):
        """Load Gated DeltaNet from flash-linear-attention (requires CUDA)."""
        try:
            from fla.layers import GatedDeltaNet
            return GatedDeltaNet(hidden_size=d_model, num_heads=state_dim)
        except ImportError as exc:
            raise ImportError(
                "flash-linear-attention (fla) is not installed. "
                "Run: pip install git+https://github.com/sustcsonglin/flash-linear-attention"
            ) from exc

    @property
    def state_bytes(self) -> int:
        """Fixed recurrent state size in bytes (backend-specific)."""
        if hasattr(self.core, "state_bytes"):
            return self.core.state_bytes
        # Fallback estimate
        return self.state_dim * self.d_model * 4

    def __call__(
        self,
        x: mx.array,
        state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass.

        Args:
            x:     Input, shape (batch, seq_len, d_model).
            state: Optional initial recurrent state.

        Returns:
            (output, final_state)
        """
        return self.core(x, state)
