"""
Unit tests for TernaryMoE and TernaryExpert.

Run with:
    pytest tests/test_ternary_moe.py -v
"""

import pytest
import mlx.core as mx

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from talh.layers.ternary_moe import TernaryMoE, TernaryExpert


# ---------------------------------------------------------------------------
# TernaryExpert
# ---------------------------------------------------------------------------

class TestTernaryExpert:
    def test_output_shape(self):
        expert = TernaryExpert(d_model=32, d_ff=64)
        x = mx.random.normal(shape=(4, 32))
        out = expert(x)
        mx.eval(out)
        assert out.shape == (4, 32)

    def test_output_is_finite(self):
        expert = TernaryExpert(d_model=16, d_ff=32)
        x = mx.random.normal(shape=(8, 16))
        out = expert(x)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()


# ---------------------------------------------------------------------------
# TernaryMoE
# ---------------------------------------------------------------------------

class TestTernaryMoE:
    @pytest.fixture
    def small_moe(self):
        return TernaryMoE(d_model=32, n_experts=4, top_k=2, d_ff=64)

    def test_output_shape(self, small_moe):
        x = mx.random.normal(shape=(2, 8, 32))
        out, aux = small_moe(x)
        mx.eval(out, aux)
        assert out.shape == (2, 8, 32)

    def test_aux_loss_is_scalar(self, small_moe):
        x = mx.random.normal(shape=(2, 8, 32))
        _, aux = small_moe(x)
        mx.eval(aux)
        assert aux.shape == () or aux.ndim == 0

    def test_aux_loss_is_finite(self, small_moe):
        x = mx.random.normal(shape=(2, 8, 32))
        _, aux = small_moe(x)
        mx.eval(aux)
        assert mx.isfinite(aux).item()

    def test_output_is_finite(self, small_moe):
        x = mx.random.normal(shape=(2, 6, 32))
        out, _ = small_moe(x)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()

    def test_top_k_constraint(self):
        """top_k > n_experts should raise an assertion."""
        with pytest.raises(AssertionError):
            TernaryMoE(d_model=16, n_experts=4, top_k=8)

    def test_memory_compression(self):
        """Ternary expert bytes should be much less than fp32 equivalent."""
        moe = TernaryMoE(d_model=64, n_experts=8, top_k=2, d_ff=256)
        assert moe.expert_bytes_ternary < moe.expert_bytes_fp32_equiv, (
            "Ternary experts should be smaller than fp32 equivalent"
        )

    def test_memory_compression_ratio(self):
        """Ternary should give ≥ 8x compression vs fp32."""
        moe = TernaryMoE(d_model=64, n_experts=8, top_k=2, d_ff=256)
        ratio = moe.expert_bytes_fp32_equiv / moe.expert_bytes_ternary
        assert ratio >= 8.0, f"Expected ≥8x compression, got {ratio:.1f}x"
