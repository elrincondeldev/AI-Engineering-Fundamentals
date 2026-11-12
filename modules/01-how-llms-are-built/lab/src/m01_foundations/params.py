"""Count the parameters, memory and compute of a decoder-only transformer, dense or mixture-of-experts.

Model cards list a parameter count, a context length and a vocabulary size. This module turns a
configuration into those numbers so you can sanity-check a card and compare models on equal terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

BYTES_PER_PARAM = {"fp32": 4.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}


@dataclass(frozen=True)
class ModelShape:
    """Hyper-parameters of a decoder-only transformer."""

    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    context_length: int
    n_kv_heads: int | None = None  # fewer K/V heads than query heads = grouped-query attention (GQA)
    ffn: Literal["mlp", "swiglu"] = "mlp"
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    attn_bias: bool = True
    ffn_bias: bool = True
    learned_positions: bool = True  # False when positions come from RoPE (no parameters)
    tied_embeddings: bool = True  # the output layer reuses the input embedding matrix
    n_experts: int = 1  # routed experts per layer; 1 means a dense feed-forward layer
    experts_per_token: int = 1
    shared_experts: int = 0  # experts every token always uses (DeepSeekMoE-style)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1


@dataclass(frozen=True)
class ParamCount:
    total: int
    active: int
    breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def non_embedding(self) -> int:
        b = self.breakdown
        return self.total - b["embeddings"] - b["positions"] - b["lm_head"]


def _norm(shape: ModelShape) -> int:
    return 2 * shape.d_model if shape.norm == "layernorm" else shape.d_model


def attention_params(s: ModelShape) -> int:
    d, kv = s.d_model, s.kv_heads * s.d_head
    weights = d * d + 2 * d * kv + d * d  # Q, K, V, output projection
    bias = (d + 2 * kv + d) if s.attn_bias else 0
    return weights + bias


def expert_params(s: ModelShape) -> int:
    """One feed-forward network (one expert)."""
    if s.ffn == "swiglu":
        return 3 * s.d_model * s.d_ff + ((2 * s.d_ff + s.d_model) if s.ffn_bias else 0)
    return 2 * s.d_model * s.d_ff + ((s.d_ff + s.d_model) if s.ffn_bias else 0)


def count_params(s: ModelShape) -> ParamCount:
    """Total parameters, parameters active for one token, and a breakdown by component."""
    expert = expert_params(s)
    experts_total = s.n_experts + s.shared_experts
    experts_active = (s.experts_per_token if s.is_moe else 1) + s.shared_experts
    router = s.d_model * s.n_experts if s.is_moe else 0
    per_layer_fixed = attention_params(s) + 2 * _norm(s) + router
    b = {
        "embeddings": s.vocab_size * s.d_model,
        "positions": s.context_length * s.d_model if s.learned_positions else 0,
        "attention": s.n_layers * attention_params(s),
        "feed_forward": s.n_layers * experts_total * expert,
        "router": s.n_layers * router,
        "norms": s.n_layers * 2 * _norm(s) + _norm(s),
        "lm_head": 0 if s.tied_embeddings else s.vocab_size * s.d_model,
    }
    total = sum(b.values())
    always = b["embeddings"] + b["positions"] + b["lm_head"] + _norm(s)
    active = always + s.n_layers * (per_layer_fixed + experts_active * expert)
    return ParamCount(total=total, active=active, breakdown=b)


def kaplan_estimate(s: ModelShape) -> int:
    """Rule of thumb from Kaplan et al. (2020): non-embedding parameters ~ 12 * n_layers * d_model^2."""
    return 12 * s.n_layers * s.d_model**2


def weight_memory_gb(n_params: int, dtype: str = "bf16") -> float:
    """Memory to hold the weights alone (no activations, no KV cache), in GB (10^9 bytes)."""
    return n_params * BYTES_PER_PARAM[dtype] / 1e9


def kv_cache_bytes(s: ModelShape, tokens: int, dtype: str = "bf16", local_window: int | None = None,
                   local_per_global: int = 0) -> float:
    """Bytes of keys and values cached for ``tokens`` tokens of context.

    With ``local_per_global = 5`` and ``local_window = 1024``, five of every six layers only keep the
    last 1024 tokens (sliding-window attention), the pattern the Gemma 3 report describes.
    """
    per_token_per_layer = 2 * s.kv_heads * s.d_head * BYTES_PER_PARAM[dtype]  # K and V
    if not local_window or not local_per_global:
        return s.n_layers * tokens * per_token_per_layer
    group = local_per_global + 1
    n_global = s.n_layers // group + (1 if s.n_layers % group > local_per_global else 0)
    n_local = s.n_layers - n_global
    return (n_global * tokens + n_local * min(tokens, local_window)) * per_token_per_layer


def training_flops(n_params: int, tokens: float) -> float:
    """Kaplan et al. estimate: about 6 floating-point operations per parameter per training token."""
    return 6.0 * n_params * tokens


# --------------------------------------------------------------------------- #
# Presets: fictional shapes chosen to illustrate design choices               #
# --------------------------------------------------------------------------- #

LAB_TINY = ModelShape("lab-tiny", vocab_size=1024, d_model=32, n_layers=2, n_heads=4, d_ff=128,
                      context_length=64, attn_bias=False, learned_positions=False)
CLASSIC_SMALL = ModelShape("classic-small", vocab_size=50257, d_model=768, n_layers=12, n_heads=12,
                           d_ff=3072, context_length=1024)
MODERN_1B = ModelShape("modern-1b", vocab_size=32000, d_model=2048, n_layers=16, n_heads=16, n_kv_heads=4,
                       d_ff=5632, context_length=8192, ffn="swiglu", norm="rmsnorm", attn_bias=False,
                       ffn_bias=False, learned_positions=False, tied_embeddings=False)
MOE_8X = replace(MODERN_1B, name="moe-8x-top2", n_experts=8, experts_per_token=2)
MOE_FINE = replace(MODERN_1B, name="moe-fine-64x-top6", d_ff=704, n_experts=64, experts_per_token=6,
                   shared_experts=2)
PRESETS = {s.name: s for s in (LAB_TINY, CLASSIC_SMALL, MODERN_1B, MOE_8X, MOE_FINE)}

def active_share(total: float, active: float) -> float:
    """Share of the parameters that one token actually uses."""
    return active / total
