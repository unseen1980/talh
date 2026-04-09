"""
Ternary Mixture of Experts (MoE) — MLX implementation.

A sparse feedforward layer where each of N experts is a TernaryLinear FFN.
For each token, a lightweight router selects the top-k experts. Only the
selected experts are activated, giving the model high parameter capacity
with low per-token compute.

Load-balancing is enforced via an auxiliary entropy loss that penalises
routing collapse (all tokens going to the same expert).

Reference:
    Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
    Mixture-of-Experts Layer", 2017.
    Ma et al., "The Era of 1-bit LLMs", 2024. (ternary experts)
"""

import mlx.core as mx
import mlx.nn as nn
from .ternary_linear import TernaryLinear


class TernaryExpert(nn.Module):
    """
    A single ternary FFN expert.

    Standard SwiGLU-style MLP with ternary weight matrices:
        FFN(x) = W_down( SiLU(W_gate(x)) * W_up(x) )

    Args:
        d_model:  Input/output dimension.
        d_ff:     Hidden dimension (typically 4 * d_model).
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = TernaryLinear(d_model, d_ff, bias=False)
        self.up   = TernaryLinear(d_model, d_ff, bias=False)
        self.down = TernaryLinear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class TernaryMoE(nn.Module):
    """
    Sparse Ternary Mixture of Experts layer.

    Routes each token to the top-k most relevant experts (by learned soft
    scores) and combines their outputs with learned weighting. All expert
    weight matrices are ternary, giving significant memory savings relative
    to a dense BF16 FFN of equivalent total parameter count.

    Args:
        d_model:       Hidden dimension.
        n_experts:     Total number of experts.
        top_k:         Number of experts activated per token.
        d_ff:          Per-expert FFN hidden dimension (default 4 * d_model).
        aux_loss_coef: Coefficient for the load-balancing auxiliary loss.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        d_ff: int | None = None,
        aux_loss_coef: float = 0.01,
    ) -> None:
        super().__init__()
        assert top_k <= n_experts, "top_k must be ≤ n_experts"

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_ff = d_ff or 4 * d_model
        self.aux_loss_coef = aux_loss_coef

        # Experts (list of TernaryExpert modules)
        self.experts = [
            TernaryExpert(d_model=d_model, d_ff=self.d_ff)
            for _ in range(n_experts)
        ]

        # Router: small dense network producing per-expert logits (FP32).
        self.router = nn.Linear(d_model, n_experts, bias=False)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Compute routing probabilities and select top-k experts.

        Args:
            x: Flattened token representations, shape (N, d_model)
               where N = batch * seq_len.

        Returns:
            (indices, weights, router_probs)
            indices:      Shape (N, top_k) — selected expert indices.
            weights:      Shape (N, top_k) — softmax-normalised scores for
                          the selected experts.
            router_probs: Shape (N, n_experts) — full softmax over all
                          experts (used for auxiliary loss).
        """
        logits = self.router(x)                       # (N, n_experts)
        router_probs = mx.softmax(logits, axis=-1)    # (N, n_experts)

        # Top-k selection
        # mlx doesn't have a built-in topk that returns both values+indices,
        # so we sort and slice.
        sorted_idx = mx.argsort(-router_probs, axis=-1)  # descending
        indices = sorted_idx[..., :self.top_k]           # (N, top_k)

        # Gather selected probabilities and renormalise
        selected_probs = mx.take_along_axis(router_probs, indices, axis=-1)
        weights = selected_probs / (mx.sum(selected_probs, axis=-1, keepdims=True) + 1e-9)

        return indices, weights, router_probs

    def _auxiliary_loss(self, router_probs: mx.array) -> mx.array:
        """
        Load-balancing auxiliary loss (entropy regularisation).

        Maximises routing entropy (uniform distribution over experts)
        to prevent collapse to a small subset of experts.

        Args:
            router_probs: Shape (N, n_experts).

        Returns:
            Scalar auxiliary loss (to be added to main loss * coef).
        """
        # Mean routing probability per expert across all tokens
        mean_probs = mx.mean(router_probs, axis=0)    # (n_experts,)
        # Negative entropy: we want to maximise entropy → minimise neg-entropy
        entropy = -mx.sum(mean_probs * mx.log(mean_probs + 1e-9))
        # We minimise -entropy (= maximise entropy)
        return -entropy * self.aux_loss_coef

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, d_model).

        Returns:
            (output, aux_loss)
            output:   Shape (batch, seq_len, d_model).
            aux_loss: Scalar load-balancing loss (add to training loss).
        """
        B, T, D = x.shape
        N = B * T

        # Flatten tokens for routing
        x_flat = x.reshape(N, D)                     # (N, d_model)

        indices, weights, router_probs = self._route(x_flat)

        # Dispatch to selected experts and combine outputs (vectorised).
        # For each (slot k, expert e): compute expert(all_tokens) then mask
        # to zero-out tokens NOT routed there. This avoids index-gather
        # operations that MLX doesn't support with a single-arg mx.where.
        output = mx.zeros((N, D))
        for k in range(self.top_k):
            expert_idx_k = indices[:, k]             # (N,) — expert index for slot k
            w_k = weights[:, k:k+1]                  # (N, 1) — routing weight for slot k

            for e_idx in range(self.n_experts):
                # Binary mask: 1.0 for tokens routed to expert e at slot k
                token_mask = (expert_idx_k == e_idx).astype(mx.float32)  # (N,)
                if not mx.any(token_mask).item():
                    continue

                # Run expert on all tokens; zero out non-selected via mask.
                y_e = self.experts[e_idx](x_flat)    # (N, d_model)
                output = output + y_e * (w_k * token_mask[:, None])

        aux_loss = self._auxiliary_loss(router_probs)
        return output.reshape(B, T, D), aux_loss

    # ------------------------------------------------------------------
    # Memory utilities
    # ------------------------------------------------------------------

    @property
    def expert_bytes_ternary(self) -> int:
        """Total ternary-packed expert parameter bytes."""
        per_expert = (
            self.d_model * self.d_ff +   # gate
            self.d_model * self.d_ff +   # up
            self.d_ff * self.d_model     # down
        )
        import math
        return math.ceil(per_expert * 2 / 8) * self.n_experts

    @property
    def expert_bytes_fp32_equiv(self) -> int:
        """What the same capacity would cost in fp32."""
        per_expert = (
            self.d_model * self.d_ff * 3  # gate + up + down
        )
        return per_expert * 4 * self.n_experts
