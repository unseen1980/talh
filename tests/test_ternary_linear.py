"""
Unit tests for TernaryLinear and quantization helpers.

Run with:
    pytest tests/test_ternary_linear.py -v
"""

import math
import pytest
import mlx.core as mx
import mlx.nn as nn

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from talh.layers.ternary_linear import (
    TernaryLinear,
    ternary_quantize,
    act_quant,
)


# ---------------------------------------------------------------------------
# ternary_quantize
# ---------------------------------------------------------------------------

class TestTernaryQuantize:
    def test_output_values_are_ternary(self):
        """All quantized values must be in {-1, 0, +1}."""
        w = mx.random.normal(shape=(64, 128))
        w_q = ternary_quantize(w)
        mx.eval(w_q)
        unique = set(w_q.flatten().tolist())
        assert unique.issubset({-1.0, 0.0, 1.0}), f"Unexpected values: {unique - {-1.0, 0.0, 1.0}}"

    def test_output_dtype(self):
        """Quantized output should be float32."""
        w = mx.random.normal(shape=(16, 32))
        w_q = ternary_quantize(w)
        assert w_q.dtype == mx.float32

    def test_shape_preserved(self):
        """Shape must be unchanged after quantization."""
        shape = (32, 64)
        w = mx.random.normal(shape=shape)
        w_q = ternary_quantize(w)
        assert w_q.shape == shape

    def test_large_values_become_nonzero(self):
        """Weights with |w| >> mean(|w|) should be ±1."""
        w = mx.array([[100.0, -100.0, 0.0001]])
        w_q = ternary_quantize(w)
        mx.eval(w_q)
        vals = w_q.flatten().tolist()
        assert vals[0] == 1.0
        assert vals[1] == -1.0
        # 0.0001 is much smaller than threshold, so it becomes 0
        assert vals[2] == 0.0

    def test_all_zeros_input(self):
        """Zero weight tensor should produce all zeros."""
        w = mx.zeros((8, 8))
        w_q = ternary_quantize(w)
        mx.eval(w_q)
        assert mx.all(w_q == 0).item()


# ---------------------------------------------------------------------------
# act_quant
# ---------------------------------------------------------------------------

class TestActQuant:
    def test_output_range(self):
        """Scaled activations must have |x| <= 1.0 + eps."""
        x = mx.random.normal(shape=(4, 128)) * 10.0
        x_scaled, scale = act_quant(x)
        mx.eval(x_scaled, scale)
        assert mx.max(mx.abs(x_scaled)).item() <= 1.0 + 1e-5

    def test_scale_shape(self):
        """Scale should broadcast over last axis (shape [..., 1])."""
        x = mx.random.normal(shape=(4, 8, 64))
        _, scale = act_quant(x)
        assert scale.shape == (4, 8, 1)

    def test_reconstruction(self):
        """x ≈ x_scaled * scale (up to numerical precision)."""
        x = mx.random.normal(shape=(2, 32))
        x_scaled, scale = act_quant(x)
        reconstructed = x_scaled * scale
        mx.eval(reconstructed)
        diff = mx.max(mx.abs(x - reconstructed)).item()
        assert diff < 1e-5, f"Reconstruction error too large: {diff}"


# ---------------------------------------------------------------------------
# TernaryLinear
# ---------------------------------------------------------------------------

class TestTernaryLinear:
    def test_output_shape(self):
        """Output shape should be (..., out_features)."""
        layer = TernaryLinear(64, 32)
        x = mx.random.normal(shape=(8, 64))
        out = layer(x)
        mx.eval(out)
        assert out.shape == (8, 32)

    def test_batched_output_shape(self):
        """Works with 3-D inputs (batch, seq, features)."""
        layer = TernaryLinear(128, 64)
        x = mx.random.normal(shape=(4, 16, 128))
        out = layer(x)
        mx.eval(out)
        assert out.shape == (4, 16, 64)

    def test_output_is_finite(self):
        """Forward pass must not produce NaN or Inf."""
        layer = TernaryLinear(32, 16)
        x = mx.random.normal(shape=(4, 32))
        out = layer(x)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()

    def test_memory_compression_vs_fp32(self):
        """Ternary packing must be smaller than fp32 baseline."""
        layer = TernaryLinear(512, 512)
        assert layer.nbytes < layer.nbytes_fp32, (
            f"Ternary bytes ({layer.nbytes}) should be < fp32 bytes ({layer.nbytes_fp32})"
        )

    def test_memory_compression_ratio(self):
        """Ternary weights should be ~16x smaller than fp32 (2-bit vs 32-bit)."""
        layer = TernaryLinear(512, 512)
        ratio = layer.nbytes_fp32 / layer.nbytes
        # 32 / 2 = 16, allow small overhead from packing rounding
        assert ratio >= 15.0, f"Expected compression ≥15x, got {ratio:.1f}x"

    def test_no_bias_option(self):
        """Layer with bias=False should not add bias to output."""
        layer_with = TernaryLinear(16, 8, bias=True)
        layer_without = TernaryLinear(16, 8, bias=False)
        assert layer_without.bias is None

    def test_quant_acts_option(self):
        """Layer with quant_acts=True should still produce correct shape."""
        layer = TernaryLinear(32, 16, quant_acts=True)
        x = mx.random.normal(shape=(4, 32))
        out = layer(x)
        mx.eval(out)
        assert out.shape == (4, 16)

    def test_ste_gradient_flows(self):
        """Gradient with respect to master weight must be non-zero (STE)."""
        layer = TernaryLinear(8, 4, bias=False)
        x = mx.random.normal(shape=(2, 8))

        def loss_fn(params):
            layer.update(params)
            out = layer(x)
            return mx.mean(out ** 2)

        loss, grads = mx.value_and_grad(loss_fn)(layer.trainable_parameters())
        mx.eval(grads)

        # At least the weight gradient should be non-zero
        w_grad = grads.get("weight", None)
        assert w_grad is not None, "No gradient for 'weight'"
        assert mx.any(w_grad != 0).item(), "Weight gradient is all zeros (STE broken)"

    def test_deterministic_quantization(self):
        """Same weights must produce same quantized values (no randomness)."""
        w = mx.random.normal(shape=(16, 16))
        q1 = ternary_quantize(w)
        q2 = ternary_quantize(w)
        mx.eval(q1, q2)
        assert mx.all(q1 == q2).item()
