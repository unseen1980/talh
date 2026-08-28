"""
TALHLayer — Parallel Hybrid Block.

Combines the SSM and MLA branches in parallel, then passes through the
Ternary MoE FFN. This is the core repeating unit of the TALH architecture.

Data flow per layer:
    x → LayerNorm
        ├── SSM Branch  → h_ssm
        └── MLA Branch  → h_mla, kv_cache
    Fusion (gated or concatenative) → h_fused
    + residual
    → LayerNorm
    → Ternary MoE → h_out
    + residual
    → output

Two fusion strategies are supported (ablation):
    'gated'  : g = σ(W_g [h_ssm; h_mla]); h = g * h_mla + (1-g) * h_ssm
    'concat' : h = W_proj [h_ssm; h_mla]

Reference:
    Dong et al., "Hymba: A Hybrid-head Architecture for Small Language
    Models", 2024. (parallel attention + SSM heads)
    Thesis Section 4.2 — Heterogeneous Parallel Routing Block.
"""

import mlx.core as mx
import mlx.nn as nn

from .mla_attention import MultiHeadLatentAttention
from .ssm_branch import SSMBranch
from .ternary_moe import TernaryMoE


class TALHLayer(nn.Module):
    """
    Single TALH transformer layer.

    Args:
        d_model:    Hidden dimension.
        num_heads:  Number of MLA attention heads.
        latent_dim: MLA KV latent dimension.
        state_dim:  SSM recurrent state size.
        n_experts:  Number of ternary MoE experts.
        top_k:      Number of experts activated per token.
        d_ff:       Per-expert FFN hidden dimension.
        fusion:     'gated' (default) or 'concat'.
        ssm_backend:'minimal', 'mamba2', or 'deltanet'.
        dropout:    Dropout probability (MLA attention).
    """

    FUSION_MODES = {"gated", "concat"}

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        latent_dim: int,
        state_dim: int = 16,
        n_experts: int = 8,
        top_k: int = 2,
        d_ff: int | None = None,
        fusion: str = "gated",
        ssm_backend: str = "minimal",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if fusion not in self.FUSION_MODES:
            raise ValueError(f"fusion must be one of {self.FUSION_MODES}, got '{fusion}'")

        self.d_model = d_model
        self.fusion = fusion

        # Pre-attention layer norm
        self.norm1 = nn.RMSNorm(d_model)
        # Pre-FFN layer norm
        self.norm2 = nn.RMSNorm(d_model)

        # --- Parallel branches ---
        self.ssm = SSMBranch(
            d_model=d_model,
            state_dim=state_dim,
            backend=ssm_backend,
        )
        self.mla = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # --- Fusion ---
        if fusion == "gated":
            # Gate network takes concatenated branches, outputs gate vector
            self.gate_proj = nn.Linear(d_model * 2, d_model, bias=False)
        else:  # concat
            # Simple linear projection from concatenated branches
            self.fuse_proj = nn.Linear(d_model * 2, d_model, bias=False)

        # --- Ternary MoE FFN ---
        self.moe = TernaryMoE(
            d_model=d_model,
            n_experts=n_experts,
            top_k=top_k,
            d_ff=d_ff,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        kv_cache: mx.array | None = None,
        ssm_state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Forward pass.

        Args:
            x:         Input hidden states, shape (batch, seq_len, d_model).
            mask:      Optional causal attention mask.
            kv_cache:  Optional MLA latent KV cache from prior steps.
            ssm_state: Optional SSM recurrent state from prior steps.

        Returns:
            (output, new_kv_cache, new_ssm_state, aux_loss)
            output:        Shape (batch, seq_len, d_model).
            new_kv_cache:  Updated MLA latent cache.
            new_ssm_state: Updated SSM recurrent state.
            aux_loss:      MoE load-balancing scalar.
        """
        residual = x

        # 1. Pre-norm
        h = self.norm1(x)

        # 2. Parallel branches
        h_ssm, new_ssm_state = self.ssm(h, state=ssm_state)
        h_mla, new_kv_cache  = self.mla(h, mask=mask, kv_cache=kv_cache)

        # 3. Fusion
        h_cat = mx.concatenate([h_ssm, h_mla], axis=-1)  # (B, T, 2*d_model)
        if self.fusion == "gated":
            g = mx.sigmoid(self.gate_proj(h_cat))         # (B, T, d_model)
            h_fused = g * h_mla + (1.0 - g) * h_ssm
        else:  # concat
            h_fused = self.fuse_proj(h_cat)               # (B, T, d_model)

        # 4. Residual connection
        x = residual + h_fused

        # 5. Pre-FFN norm
        h = self.norm2(x)

        # 6. Ternary MoE
        h_moe, aux_loss = self.moe(h)

        # 7. Residual connection
        output = x + h_moe

        return output, new_kv_cache, new_ssm_state, aux_loss
