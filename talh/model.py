"""
TALH Full Model — MLX implementation.

Stacks N TALHLayer blocks with token embedding and language model head.
Supports both full-sequence (training/prefill) and incremental (decode)
modes via the kv_caches / ssm_states arguments.

Architecture (per layer):
    Embedding → [TALHLayer × n_layers] → RMSNorm → LM Head (tied weights)

Usage (MLX / Mac):
    from talh.model import TALH, TALHConfig
    cfg = TALHConfig(vocab_size=32000, n_layers=4, d_model=256, ...)
    model = TALH(cfg)
    logits, caches, states = model(input_ids)
"""

from dataclasses import dataclass, field
import mlx.core as mx
import mlx.nn as nn

from .layers.talh_layer import TALHLayer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TALHConfig:
    """
    Hyperparameter configuration for the TALH model.

    Default values produce a ~15M parameter "nano" model suitable for
    TinyStories training and hypothesis validation on Mac/Colab.
    """

    vocab_size:    int   = 32000
    n_layers:      int   = 4
    d_model:       int   = 256
    num_heads:     int   = 4       # MLA attention heads
    latent_dim:    int   = 64      # MLA KV latent dimension
    state_dim:     int   = 16      # SSM recurrent state size
    n_experts:     int   = 8       # MoE total experts
    top_k:         int   = 2       # MoE active experts per token
    d_ff:          int   = 512     # Per-expert FFN hidden dim (default 2*d_model)
    fusion:        str   = "gated" # 'gated' or 'concat'
    ssm_backend:   str   = "minimal"
    dropout:       float = 0.0
    max_seq_len:   int   = 4096


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TALH(nn.Module):
    """
    Ternary Adaptive Latent Hybrid language model.

    Args:
        config: TALHConfig instance.
    """

    def __init__(self, config: TALHConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = [
            TALHLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                latent_dim=config.latent_dim,
                state_dim=config.state_dim,
                n_experts=config.n_experts,
                top_k=config.top_k,
                d_ff=config.d_ff,
                fusion=config.fusion,
                ssm_backend=config.ssm_backend,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ]

        self.norm = nn.RMSNorm(config.d_model)
        # LM head — weight tied to embedding for parameter efficiency
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        mask: mx.array | None = None,
        kv_caches: list[mx.array] | None = None,
        ssm_states: list[mx.array] | None = None,
    ) -> tuple[mx.array, list[mx.array], list[mx.array], mx.array]:
        """
        Forward pass.

        Args:
            input_ids:  Token ids, shape (batch, seq_len).
            mask:       Optional causal attention mask.
            kv_caches:  List of MLA latent caches, one per layer.
                        Pass None for fresh inference (no cached context).
            ssm_states: List of SSM recurrent states, one per layer.
                        Pass None for fresh inference.

        Returns:
            (logits, new_kv_caches, new_ssm_states, total_aux_loss)
            logits:          Shape (batch, seq_len, vocab_size).
            new_kv_caches:   Updated per-layer MLA caches.
            new_ssm_states:  Updated per-layer SSM states.
            total_aux_loss:  Sum of per-layer MoE load-balancing losses.
        """
        B, T = input_ids.shape

        # 1. Token embedding
        x = self.embed(input_ids)  # (B, T, d_model)

        # 2. Causal mask (if not provided)
        if mask is None and T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

        # 3. Per-layer caches (None = fresh)
        if kv_caches is None:
            kv_caches = [None] * self.config.n_layers
        if ssm_states is None:
            ssm_states = [None] * self.config.n_layers

        # 4. Layer stack
        new_kv_caches = []
        new_ssm_states = []
        total_aux_loss = mx.zeros(())

        for i, layer in enumerate(self.layers):
            x, new_kv, new_state, aux = layer(
                x,
                mask=mask,
                kv_cache=kv_caches[i],
                ssm_state=ssm_states[i],
            )
            new_kv_caches.append(new_kv)
            new_ssm_states.append(new_state)
            total_aux_loss = total_aux_loss + aux

        # 5. Final norm + LM head
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, new_kv_caches, new_ssm_states, total_aux_loss

    # ------------------------------------------------------------------
    # Convenience: parameter count
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        """Total parameter count (all weights, not just active ones)."""
        total = 0
        for _, v in self.trainable_parameters().items() if hasattr(self.trainable_parameters(), "items") else []:
            if isinstance(v, mx.array):
                total += v.size
        return total

    def memory_summary(self) -> dict:
        """
        Rough memory breakdown by component type.

        Returns a dict with keys: 'embedding_mb', 'lm_head_mb',
        'mla_mb', 'ssm_mb', 'moe_mb', 'total_mb'.
        """
        cfg = self.config
        embed_params  = cfg.vocab_size * cfg.d_model
        mla_per_layer = (
            cfg.d_model * cfg.d_model +   # q_proj
            cfg.d_model * cfg.latent_dim + # kv_down
            cfg.latent_dim * cfg.d_model + # k_up
            cfg.latent_dim * cfg.d_model + # v_up
            cfg.d_model * cfg.d_model      # out_proj
        )
        ssm_per_layer  = (
            cfg.d_model * cfg.d_model * 2 +  # in_proj
            cfg.d_model * cfg.state_dim * 3 + # A,B,C
            cfg.d_model * cfg.d_model          # out_proj
        )
        import math
        # MoE: ternary experts (2 bits/weight)
        expert_weights = cfg.d_model * cfg.d_ff * 3 * cfg.n_experts
        moe_bytes = math.ceil(expert_weights * 2 / 8) * cfg.n_layers
        moe_mb    = moe_bytes / 1e6

        bf16_bytes = 2
        return {
            "embedding_mb": embed_params * bf16_bytes / 1e6,
            "lm_head_mb":   embed_params * bf16_bytes / 1e6,
            "mla_mb":       mla_per_layer * cfg.n_layers * bf16_bytes / 1e6,
            "ssm_mb":       ssm_per_layer * cfg.n_layers * bf16_bytes / 1e6,
            "moe_mb":       moe_mb,
            "total_mb":     (
                embed_params * 2 * bf16_bytes +
                mla_per_layer * cfg.n_layers * bf16_bytes +
                ssm_per_layer * cfg.n_layers * bf16_bytes
            ) / 1e6 + moe_mb,
        }
