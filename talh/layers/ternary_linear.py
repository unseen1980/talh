"""
Ternary Linear Layer — MLX implementation.

Weights are constrained to {-1, 0, +1} (1.58-bit parameterization) using
quantization-aware training (QAT) with a straight-through estimator (STE)
for gradient flow through the quantization step.

Master weights are kept in float32 for stable optimization; quantization
is applied only in the forward pass. This matches the BitNet b1.58 recipe
described in Section 3.1 of the TALH thesis.

Reference:
    Ma et al., "The Era of 1-bit LLMs: All Large Language Models are
    in 1.58 Bits", 2024.
"""

import math
import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def act_quant(x: mx.array, eps: float = 1e-6) -> tuple[mx.array, mx.array]:
    """
    Absmax activation quantization to [-1, 1].

    Args:
        x:   Input activations, shape (..., d).
        eps: Small constant for numerical stability.

    Returns:
        (x_scaled, scale) where x_scaled = x / scale and
        scale = max(|x|) along the last axis, broadcastable.
    """
    scale = mx.maximum(
        mx.max(mx.abs(x), axis=-1, keepdims=True),
        eps,
    )
    return x / scale, scale


def ternary_quantize(w: mx.array) -> mx.array:
    """
    Threshold-based ternary quantization: maps weights to {-1, 0, +1}.

    Threshold τ = 0.5 * mean(|w|) as described in the thesis.
    Values with |w| > τ are mapped to ±1; the rest become 0.

    Args:
        w: Weight tensor of any shape (float32).

    Returns:
        Ternary weight tensor in {-1.0, 0.0, +1.0} as float32.
    """
    thresh = 0.5 * mx.mean(mx.abs(w))
    pos = (w > thresh).astype(mx.float32)
    neg = (w < -thresh).astype(mx.float32)
    return pos - neg


# ---------------------------------------------------------------------------
# TernaryLinear module
# ---------------------------------------------------------------------------

class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weight quantization (QAT / STE).

    In the forward pass the master float32 weight is quantized to {-1,0,1}
    before the matrix multiply. During backprop MLX's automatic
    differentiation treats the quantization as an identity (straight-through
    estimator), so gradients flow to the master weight unchanged.

    An optional residual scale γ (learnable scalar, initialised to 1) is
    applied to the output for training stability, following common practice
    in low-bit networks.

    Args:
        in_features:  Size of each input sample.
        out_features: Size of each output sample.
        bias:         If True, adds a learnable bias.
        quant_acts:   If True, also quantizes input activations via absmax.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_acts: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_acts = quant_acts

        # Master weights — kept in float32 for stable gradient updates.
        scale = 1.0 / math.sqrt(in_features)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(out_features, in_features)
        )

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

        # Learnable residual scale for output stability.
        self.gamma = mx.ones((1,))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nbytes(self) -> int:
        """Effective bytes occupied by ternary weights (packed 2-bit)."""
        # Each ternary value needs 2 bits; master weights not counted here.
        n_params = self.out_features * self.in_features
        return math.ceil(n_params * 2 / 8)

    @property
    def nbytes_fp32(self) -> int:
        """Bytes if weights were stored as float32 (comparison baseline)."""
        return self.out_features * self.in_features * 4

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (..., in_features).

        Returns:
            Output tensor, shape (..., out_features).
        """
        # 1. Optionally quantize activations.
        if self.quant_acts:
            x, _scale = act_quant(x)

        # 2. Quantize weights to {-1, 0, +1} — STE applied implicitly by
        #    MLX autograd (no mx.stop_gradient needed; quantize is treated
        #    as identity for gradient purposes via the standard STE trick:
        #    we compute the quantized weight but use the master weight for
        #    the gradient path by adding zero: w_q = w + (w_q - w).stop()).
        w_q = ternary_quantize(self.weight)
        # STE: gradients pass through as if w_q == self.weight
        w_q = self.weight + mx.stop_gradient(w_q - self.weight)

        # 3. Linear transform.
        out = x @ w_q.T

        # 4. Residual scale.
        out = out * self.gamma

        # 5. Bias.
        if self.bias is not None:
            out = out + self.bias

        return out
