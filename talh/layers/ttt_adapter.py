"""
Gated Test-Time Training (TTT) Adapter — MLX implementation.

Provides inference-time adaptation via LoRA-style low-rank parameter
updates. Updates are triggered only when a "surprise signal" exceeds a
threshold, keeping the adapter inactive (and cost-free) for most tokens.

Safety constraints prevent drift:
  - Updates are low-rank (bounded adaptation subspace).
  - A reset policy clears deltas at session boundaries.
  - An adaptation budget limits the total number of updates per session.

Reference:
    Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive
    Hidden States", 2024.
    Zhang et al., "Large Chunk Test-Time Training", 2025.
    Akyürek et al., "The Surprising Effectiveness of Test-Time Training
    for Abstract Reasoning", 2025.
"""

import mlx.core as mx
import mlx.nn as nn


class LoRADelta(nn.Module):
    """
    Low-rank parameter delta: Δ = B @ A where A ∈ R^{r×d}, B ∈ R^{d×r}.

    The effective weight update is W' = W + scale * B @ A.
    At initialisation B = 0 so the adapter is a no-op.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        rank:         LoRA rank (r ≪ in_features, out_features).
        scale:        Learning-rate-style scaling factor (α/r in the paper).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scale = scale / rank

        # A is randomly initialised (Kaiming); B is zero → no-op at start.
        self.A = mx.random.normal(shape=(rank, in_features)) * (in_features ** -0.5)
        self.B = mx.zeros((out_features, rank))

    def __call__(self, x: mx.array) -> mx.array:
        """Compute low-rank delta output: x → x @ A.T @ B.T * scale."""
        return x @ self.A.T @ self.B.T * self.scale

    def reset(self) -> None:
        """Zero the B matrix — restores the adapter to a no-op."""
        self.B = mx.zeros_like(self.B)

    @property
    def nbytes(self) -> int:
        """Bytes used by adapter parameters (float32)."""
        return (self.A.size + self.B.size) * 4


class SurpriseDetector:
    """
    Stateful detector for adaptation trigger signals.

    Tracks a rolling mean of next-token loss and fires when the current
    loss spikes above threshold * rolling_mean.

    Args:
        loss_threshold: Multiplier above rolling mean to trigger (e.g. 2.0).
        momentum:       EMA momentum for the rolling mean (0 < momentum < 1).
    """

    def __init__(
        self,
        loss_threshold: float = 2.0,
        momentum: float = 0.99,
    ) -> None:
        self.loss_threshold = loss_threshold
        self.momentum = momentum
        self._rolling_mean: float | None = None

    def update(self, loss: float) -> bool:
        """
        Update the rolling mean and return True if a trigger fires.

        Args:
            loss: Current per-token loss value.

        Returns:
            True if adaptation should be triggered.
        """
        if self._rolling_mean is None:
            self._rolling_mean = loss
            return False

        triggered = loss > self.loss_threshold * self._rolling_mean
        self._rolling_mean = (
            self.momentum * self._rolling_mean + (1 - self.momentum) * loss
        )
        return triggered

    def reset(self) -> None:
        """Reset rolling mean (call at session boundary)."""
        self._rolling_mean = None


class GatedTTTAdapter(nn.Module):
    """
    Gated TTT Adapter applied to a single linear projection layer.

    Wraps a base linear layer with a LoRA delta that is updated in-place
    during inference when the surprise detector fires.

    The adapter is gated (not always-on) to minimise interference with
    the base model's learned representations.

    Args:
        base_layer:       The nn.Linear (or TernaryLinear) to adapt.
        rank:             LoRA rank for the adaptation delta.
        lr:               Learning rate for in-place gradient update.
        loss_threshold:   Surprise threshold multiplier.
        momentum:         EMA momentum for rolling loss mean.
        budget:           Maximum number of adaptation steps per session.
                          Prevents unbounded drift.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 4,
        lr: float = 1e-4,
        loss_threshold: float = 2.0,
        momentum: float = 0.99,
        budget: int = 100,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lr = lr
        self.budget = budget
        self._steps = 0

        # Infer dimensions from the base layer
        if hasattr(base_layer, "weight"):
            out_features, in_features = base_layer.weight.shape
        else:
            raise AttributeError("base_layer must have a .weight attribute")

        self.lora = LoRADelta(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
        )
        self.detector = SurpriseDetector(
            loss_threshold=loss_threshold,
            momentum=momentum,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass: base output + LoRA delta.

        Args:
            x: Input, shape (..., in_features).

        Returns:
            Output, shape (..., out_features).
        """
        base_out = self.base_layer(x)
        delta_out = self.lora(x)
        return base_out + delta_out

    # ------------------------------------------------------------------
    # In-place adaptation
    # ------------------------------------------------------------------

    def adapt(self, x: mx.array, target_loss: float) -> bool:
        """
        Conditionally update LoRA delta based on surprise signal.

        Should be called after each forward pass with the per-token loss.
        If the surprise detector fires AND the adaptation budget is not
        exhausted, performs one gradient step on the LoRA B matrix.

        Args:
            x:           Input that produced the most recent output.
            target_loss: Per-token cross-entropy loss for the last step.

        Returns:
            True if an adaptation step was taken.
        """
        if self._steps >= self.budget:
            return False
        if not self.detector.update(target_loss):
            return False

        # Compute gradient of loss w.r.t. LoRA B only.
        def lora_loss(params):
            self.lora.update(params)
            out = self(x)
            return mx.mean(out ** 2)   # proxy; replace with real LM loss in trainer

        grads = mx.grad(lora_loss)({"B": self.lora.B})
        B_grad = grads.get("B")
        if B_grad is not None:
            self.lora.B = self.lora.B - self.lr * B_grad
            mx.eval(self.lora.B)

        self._steps += 1
        return True

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """
        Reset adapter state for a new session.

        Clears LoRA delta, surprise detector state, and step counter.
        Call this between independent user sessions.
        """
        self.lora.reset()
        self.detector.reset()
        self._steps = 0

    @property
    def nbytes(self) -> int:
        """Bytes used by the LoRA adapter parameters."""
        return self.lora.nbytes
