"""
Unit tests for TALHLayer (parallel hybrid block).

Run with:
    pytest tests/test_talh_layer.py -v
"""

import pytest
import mlx.core as mx

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from talh.layers.talh_layer import TALHLayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_layer(fusion="gated"):
    return TALHLayer(
        d_model=64,
        num_heads=4,
        latent_dim=16,
        state_dim=8,
        n_experts=4,
        top_k=2,
        d_ff=128,
        fusion=fusion,
    )


# ---------------------------------------------------------------------------
# Shape & dtype
# ---------------------------------------------------------------------------

class TestTALHLayerShape:
    def test_output_shape_gated(self):
        layer = make_layer(fusion="gated")
        x = mx.random.normal(shape=(2, 8, 64))
        out, kv, state, aux = layer(x)
        mx.eval(out)
        assert out.shape == (2, 8, 64)

    def test_output_shape_concat(self):
        layer = make_layer(fusion="concat")
        x = mx.random.normal(shape=(2, 8, 64))
        out, kv, state, aux = layer(x)
        mx.eval(out)
        assert out.shape == (2, 8, 64)

    def test_kv_cache_shape(self):
        layer = make_layer()
        x = mx.random.normal(shape=(2, 8, 64))
        _, kv, _, _ = layer(x)
        mx.eval(kv)
        assert kv.shape == (2, 8, 16)  # (batch, seq, latent_dim)

    def test_ssm_state_shape_fixed(self):
        """SSM state must have the same shape regardless of sequence length."""
        layer = make_layer()
        x4   = mx.random.normal(shape=(1,   4, 64))
        x128 = mx.random.normal(shape=(1, 128, 64))
        _, _, s4,   _ = layer(x4)
        _, _, s128, _ = layer(x128)
        mx.eval(s4, s128)
        assert s4.shape == s128.shape

    def test_aux_loss_scalar(self):
        layer = make_layer()
        x = mx.random.normal(shape=(2, 6, 64))
        _, _, _, aux = layer(x)
        mx.eval(aux)
        assert aux.ndim == 0 or aux.shape == ()

    def test_output_is_finite(self):
        layer = make_layer()
        x = mx.random.normal(shape=(2, 6, 64))
        out, _, _, _ = layer(x)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()


# ---------------------------------------------------------------------------
# Incremental (cached) inference
# ---------------------------------------------------------------------------

class TestTALHLayerIncremental:
    def test_kv_cache_grows_across_steps(self):
        layer = make_layer()
        x1 = mx.random.normal(shape=(1, 4, 64))
        _, kv1, st1, _ = layer(x1)
        mx.eval(kv1, st1)
        assert kv1.shape[1] == 4

        x2 = mx.random.normal(shape=(1, 4, 64))
        _, kv2, _, _ = layer(x2, kv_cache=kv1, ssm_state=st1)
        mx.eval(kv2)
        assert kv2.shape[1] == 8  # 4 + 4


# ---------------------------------------------------------------------------
# Fusion modes
# ---------------------------------------------------------------------------

class TestFusionModes:
    def test_invalid_fusion_raises(self):
        with pytest.raises(ValueError, match="fusion must be one of"):
            TALHLayer(
                d_model=32, num_heads=2, latent_dim=8,
                fusion="bad_mode",
            )

    def test_gated_fusion_gate_not_trivial(self):
        """Gated fusion gates should not all be 0 or 1 (not degenerate)."""
        layer = make_layer(fusion="gated")
        x = mx.random.normal(shape=(2, 16, 64))
        # Forward through norm + branches + gate directly
        h = layer.norm1(x)
        h_ssm, _ = layer.ssm(h)
        h_mla, _ = layer.mla(h)
        h_cat = mx.concatenate([h_ssm, h_mla], axis=-1)
        g = mx.sigmoid(layer.gate_proj(h_cat))
        mx.eval(g)
        g_vals = g.flatten().tolist()
        # Should have values that aren't all 0 or all 1
        has_intermediate = any(0.05 < v < 0.95 for v in g_vals)
        assert has_intermediate, "Gate values are trivially saturated"
