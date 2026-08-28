"""
Unit tests for GatedTTTAdapter, LoRADelta, and SurpriseDetector.

Run with:
    pytest tests/test_ttt_adapter.py -v
"""

import pytest
import mlx.core as mx
import mlx.nn as nn

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from talh.layers.ttt_adapter import GatedTTTAdapter, LoRADelta, SurpriseDetector


# ---------------------------------------------------------------------------
# LoRADelta
# ---------------------------------------------------------------------------

class TestLoRADelta:
    def test_output_shape(self):
        lora = LoRADelta(in_features=16, out_features=32, rank=4)
        x = mx.random.normal(shape=(4, 16))
        out = lora(x)
        mx.eval(out)
        assert out.shape == (4, 32)

    def test_initial_output_is_zero(self):
        """At init B=0, so output should be all zeros."""
        lora = LoRADelta(in_features=16, out_features=32, rank=4)
        x = mx.random.normal(shape=(4, 16))
        out = lora(x)
        mx.eval(out)
        assert mx.max(mx.abs(out)).item() < 1e-7, "Initial LoRA output should be zero"

    def test_reset_zeroes_output(self):
        """After reset, output should be zero again."""
        lora = LoRADelta(in_features=8, out_features=16, rank=2)
        # Manually set B to non-zero
        lora.B = mx.ones_like(lora.B)
        lora.reset()
        x = mx.random.normal(shape=(2, 8))
        out = lora(x)
        mx.eval(out)
        assert mx.max(mx.abs(out)).item() < 1e-7

    def test_nbytes_positive(self):
        lora = LoRADelta(in_features=64, out_features=64, rank=8)
        assert lora.nbytes > 0


# ---------------------------------------------------------------------------
# SurpriseDetector
# ---------------------------------------------------------------------------

class TestSurpriseDetector:
    def test_no_trigger_on_first_call(self):
        """First call always returns False (no baseline yet)."""
        det = SurpriseDetector(loss_threshold=2.0)
        triggered = det.update(1.0)
        assert not triggered

    def test_triggers_on_spike(self):
        """A large spike above threshold should trigger."""
        det = SurpriseDetector(loss_threshold=2.0, momentum=0.0)
        det.update(1.0)       # establish baseline
        triggered = det.update(10.0)   # 10x spike
        assert triggered

    def test_no_trigger_on_normal_loss(self):
        """Losses near the baseline should not trigger."""
        det = SurpriseDetector(loss_threshold=2.0, momentum=0.5)
        det.update(1.0)
        triggered = det.update(1.1)
        assert not triggered

    def test_reset_clears_state(self):
        """After reset, first call should again return False."""
        det = SurpriseDetector(loss_threshold=2.0)
        det.update(1.0)
        det.reset()
        triggered = det.update(100.0)
        assert not triggered


# ---------------------------------------------------------------------------
# GatedTTTAdapter
# ---------------------------------------------------------------------------

class TestGatedTTTAdapter:
    @pytest.fixture
    def adapter(self):
        base = nn.Linear(16, 32)
        return GatedTTTAdapter(base_layer=base, rank=4, budget=10)

    def test_output_shape(self, adapter):
        x = mx.random.normal(shape=(2, 16))
        out = adapter(x)
        mx.eval(out)
        assert out.shape == (2, 32)

    def test_initial_output_equals_base(self, adapter):
        """Before any adaptation, output should equal base layer output."""
        x = mx.random.normal(shape=(2, 16))
        base_out = adapter.base_layer(x)
        adapter_out = adapter(x)
        mx.eval(base_out, adapter_out)
        diff = mx.max(mx.abs(base_out - adapter_out)).item()
        assert diff < 1e-6, f"Initial adapter output differs from base: {diff}"

    def test_nbytes_positive(self, adapter):
        assert adapter.nbytes > 0

    def test_budget_respected(self):
        """Adapter should not exceed budget adaptation steps."""
        base = nn.Linear(8, 16)
        adapter = GatedTTTAdapter(base_layer=base, rank=2, budget=3, loss_threshold=0.1)
        x = mx.random.normal(shape=(2, 8))

        steps_taken = 0
        # Force many triggers with high loss
        for _ in range(20):
            took_step = adapter.adapt(x, target_loss=100.0)
            if took_step:
                steps_taken += 1

        assert steps_taken <= 3, f"Exceeded budget: {steps_taken} steps taken"

    def test_reset_session_resets_steps(self):
        """After reset_session, budget counter should restart."""
        base = nn.Linear(8, 16)
        adapter = GatedTTTAdapter(base_layer=base, rank=2, budget=2, loss_threshold=0.1)
        x = mx.random.normal(shape=(2, 8))

        # Exhaust budget
        for _ in range(10):
            adapter.adapt(x, target_loss=100.0)
        assert adapter._steps >= 2

        adapter.reset_session()
        assert adapter._steps == 0

    def test_missing_weight_attribute_raises(self):
        """Base layer without .weight should raise AttributeError."""

        class NoWeightLayer(nn.Module):
            def __call__(self, x):
                return x

        with pytest.raises(AttributeError, match="weight"):
            GatedTTTAdapter(base_layer=NoWeightLayer())
