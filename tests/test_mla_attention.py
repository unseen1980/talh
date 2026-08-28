"""
Unit tests for MultiHeadLatentAttention.

Run with:
    pytest tests/test_mla_attention.py -v
"""

import pytest
import mlx.core as mx

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from talh.layers.mla_attention import MultiHeadLatentAttention


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_mla():
    return MultiHeadLatentAttention(d_model=64, num_heads=4, latent_dim=16)


# ---------------------------------------------------------------------------
# Shape & dtype
# ---------------------------------------------------------------------------

class TestMLAShape:
    def test_output_shape(self, small_mla):
        x = mx.random.normal(shape=(2, 8, 64))
        out, cache = small_mla(x)
        mx.eval(out, cache)
        assert out.shape == (2, 8, 64)

    def test_cache_shape(self, small_mla):
        """Cache should be (batch, seq_len, latent_dim)."""
        x = mx.random.normal(shape=(2, 8, 64))
        _, cache = small_mla(x)
        mx.eval(cache)
        assert cache.shape == (2, 8, 16)

    def test_incremental_cache_grows(self, small_mla):
        """Cache length should grow by seq_len on each call."""
        x1 = mx.random.normal(shape=(1, 4, 64))
        _, cache1 = small_mla(x1)
        mx.eval(cache1)
        assert cache1.shape[1] == 4

        x2 = mx.random.normal(shape=(1, 4, 64))
        _, cache2 = small_mla(x2, kv_cache=cache1)
        mx.eval(cache2)
        assert cache2.shape[1] == 8  # 4 + 4

    def test_output_is_finite(self, small_mla):
        x = mx.random.normal(shape=(2, 6, 64))
        out, _ = small_mla(x)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()


# ---------------------------------------------------------------------------
# KV cache compression (H1 evidence)
# ---------------------------------------------------------------------------

class TestKVCacheCompression:
    def test_mla_cache_smaller_than_standard_mha(self):
        """MLA cache bytes should be << standard MHA cache bytes."""
        mla = MultiHeadLatentAttention(d_model=512, num_heads=8, latent_dim=64)
        seq_len = 8192

        mla_bytes = mla.kv_cache_bytes(seq_len=seq_len, dtype_bytes=2)
        mha_bytes = MultiHeadLatentAttention.standard_mha_cache_bytes(
            seq_len=seq_len,
            num_heads=8,
            head_dim=64,
            dtype_bytes=2,
        )
        assert mla_bytes < mha_bytes, (
            f"MLA cache ({mla_bytes}B) should be < standard MHA cache ({mha_bytes}B)"
        )

    def test_compression_ratio_significant(self):
        """Compression ratio should be ≥ 8x (thesis targets ~93% reduction)."""
        mla = MultiHeadLatentAttention(d_model=512, num_heads=8, latent_dim=64)
        ratio = mla.compression_ratio(dtype_bytes=2)
        assert ratio >= 8.0, f"Expected ≥8x compression, got {ratio:.1f}x"

    def test_cache_linear_growth(self):
        """Cache size should grow linearly with sequence length."""
        mla = MultiHeadLatentAttention(d_model=128, num_heads=4, latent_dim=16)
        b8 = mla.kv_cache_bytes(seq_len=8)
        b16 = mla.kv_cache_bytes(seq_len=16)
        assert b16 == 2 * b8


# ---------------------------------------------------------------------------
# Compress / expand round-trip
# ---------------------------------------------------------------------------

class TestCompressExpand:
    def test_compress_shape(self, small_mla):
        x = mx.random.normal(shape=(2, 10, 64))
        latent = small_mla.compress_kv(x)
        mx.eval(latent)
        assert latent.shape == (2, 10, 16)

    def test_expand_shape(self, small_mla):
        latent = mx.random.normal(shape=(2, 10, 16))
        k, v = small_mla.expand_kv(latent)
        mx.eval(k, v)
        assert k.shape == (2, 10, 64)
        assert v.shape == (2, 10, 64)

    def test_expand_output_finite(self, small_mla):
        latent = mx.random.normal(shape=(2, 10, 16))
        k, v = small_mla.expand_kv(latent)
        mx.eval(k, v)
        assert mx.all(mx.isfinite(k)).item()
        assert mx.all(mx.isfinite(v)).item()
