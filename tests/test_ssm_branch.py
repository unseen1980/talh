"""
Unit tests for SSMBranch (MinimalSSM backend).

Run with:
    pytest tests/test_ssm_branch.py -v
"""

import pytest
import mlx.core as mx

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from talh.layers.ssm_branch import SSMBranch, MinimalSSM


# ---------------------------------------------------------------------------
# MinimalSSM
# ---------------------------------------------------------------------------

class TestMinimalSSM:
    def test_output_shape(self):
        ssm = MinimalSSM(d_model=32, state_dim=8)
        x = mx.random.normal(shape=(2, 10, 32))
        out, state = ssm(x)
        mx.eval(out, state)
        assert out.shape == (2, 10, 32)

    def test_state_shape(self):
        ssm = MinimalSSM(d_model=32, state_dim=8, expand=2)
        x = mx.random.normal(shape=(2, 10, 32))
        _, state = ssm(x)
        mx.eval(state)
        # state: (batch, d_inner, state_dim) = (2, 64, 8)
        assert state.shape == (2, 64, 8)

    def test_state_size_is_fixed(self):
        """
        State size must NOT grow with sequence length (H4 core property).
        """
        ssm = MinimalSSM(d_model=16, state_dim=4)
        x_short = mx.random.normal(shape=(1, 8, 16))
        x_long  = mx.random.normal(shape=(1, 256, 16))
        _, state_short = ssm(x_short)
        _, state_long  = ssm(x_long)
        mx.eval(state_short, state_long)
        assert state_short.shape == state_long.shape, (
            "SSM state size must be independent of sequence length"
        )

    def test_output_is_finite(self):
        ssm = MinimalSSM(d_model=16, state_dim=4)
        x = mx.random.normal(shape=(2, 12, 16))
        out, _ = ssm(x)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()

    def test_state_carries_over(self):
        """Passing the previous state should change outputs (stateful)."""
        ssm = MinimalSSM(d_model=16, state_dim=4)
        x = mx.random.normal(shape=(1, 4, 16))

        out1, state1 = ssm(x)
        out2, _      = ssm(x, state=None)
        out3, _      = ssm(x, state=state1)
        mx.eval(out1, out2, out3)

        # out2 (fresh state) should equal out1; out3 (warm state) should differ
        diff_fresh = mx.max(mx.abs(out1 - out2)).item()
        diff_warm  = mx.max(mx.abs(out1 - out3)).item()
        assert diff_fresh < 1e-5, "Same input + fresh state → same output"
        assert diff_warm  > 0.0,  "Same input + warm state  → different output"


# ---------------------------------------------------------------------------
# SSMBranch (public interface)
# ---------------------------------------------------------------------------

class TestSSMBranch:
    def test_minimal_backend_output_shape(self):
        branch = SSMBranch(d_model=32, state_dim=8, backend="minimal")
        x = mx.random.normal(shape=(2, 10, 32))
        out, state = branch(x)
        mx.eval(out, state)
        assert out.shape == (2, 10, 32)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend must be one of"):
            SSMBranch(d_model=32, state_dim=8, backend="unknown")

    def test_state_bytes_positive(self):
        branch = SSMBranch(d_model=32, state_dim=8)
        assert branch.state_bytes > 0

    def test_state_fixed_regardless_of_seq_len(self):
        """SSMBranch state must have the same shape for T=8 and T=512."""
        branch = SSMBranch(d_model=16, state_dim=4)
        _, s8   = branch(mx.random.normal(shape=(1,   8, 16)))
        _, s512 = branch(mx.random.normal(shape=(1, 512, 16)))
        mx.eval(s8, s512)
        assert s8.shape == s512.shape
