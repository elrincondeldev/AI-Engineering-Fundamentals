"""A decoder-only transformer forward pass in plain numpy.

Shapes use T for sequence length, D for model width, H for heads and V for vocabulary size.
Weights are random (seeded): the point is to see every operation and every shape, not to train.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class TinyConfig:
    vocab_size: int = 1024
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    context_length: int = 64

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


def softmax(x: Array, axis: int = -1) -> Array:
    """Numerically stable softmax: subtract the max before exponentiating."""
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x: Array, gain: Array, bias: Array, eps: float = 1e-5) -> Array:
    """Normalise each position's vector to mean 0 and variance 1, then rescale."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gain * (x - mean) / np.sqrt(var + eps) + bias


def gelu(x: Array) -> Array:
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def causal_mask(t: int) -> Array:
    """Boolean (T, T) matrix: True where position i may look at position j (j <= i)."""
    return np.tril(np.ones((t, t), dtype=bool))


def rope(x: Array, positions: Array, base: float = 10000.0) -> Array:
    """Rotary position embedding: rotate each pair of dimensions by an angle proportional to position.

    ``x`` has shape (..., T, d) with d even. After rotation, the dot product of a query at position
    m and a key at position n depends only on m - n.
    """
    d = x.shape[-1]
    freqs = base ** (-np.arange(0, d, 2) / d)  # (d/2,)
    angles = positions[:, None] * freqs[None, :]  # (T, d/2)
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def scaled_dot_product_attention(
    q: Array, k: Array, v: Array, mask: Array | None = None
) -> tuple[Array, Array]:
    """softmax(Q K^T / sqrt(d)) V. Returns (output, attention weights)."""
    d = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / math.sqrt(d)  # (..., T, T)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


def init_params(cfg: TinyConfig, seed: int = 0) -> dict[str, Array]:
    """Random weights with a small standard deviation, as at the start of training."""
    rng = np.random.default_rng(seed)
    std = 0.02
    p: dict[str, Array] = {"tok_emb": rng.normal(0, std, (cfg.vocab_size, cfg.d_model))}
    for i in range(cfg.n_layers):
        p[f"l{i}.ln1.g"], p[f"l{i}.ln1.b"] = np.ones(cfg.d_model), np.zeros(cfg.d_model)
        for name in ("wq", "wk", "wv", "wo"):
            p[f"l{i}.{name}"] = rng.normal(0, std, (cfg.d_model, cfg.d_model))
        p[f"l{i}.ln2.g"], p[f"l{i}.ln2.b"] = np.ones(cfg.d_model), np.zeros(cfg.d_model)
        p[f"l{i}.w1"] = rng.normal(0, std, (cfg.d_model, cfg.d_ff))
        p[f"l{i}.b1"] = np.zeros(cfg.d_ff)
        p[f"l{i}.w2"] = rng.normal(0, std, (cfg.d_ff, cfg.d_model))
        p[f"l{i}.b2"] = np.zeros(cfg.d_model)
    p["ln_f.g"], p["ln_f.b"] = np.ones(cfg.d_model), np.zeros(cfg.d_model)
    return p  # the output layer reuses tok_emb ("tied embeddings")


def split_heads(x: Array, n_heads: int) -> Array:
    """(T, D) -> (H, T, D/H)"""
    t, d = x.shape
    return x.reshape(t, n_heads, d // n_heads).transpose(1, 0, 2)


def merge_heads(x: Array) -> Array:
    """(H, T, D/H) -> (T, D)"""
    h, t, dh = x.shape
    return x.transpose(1, 0, 2).reshape(t, h * dh)


def multi_head_attention(x: Array, p: dict[str, Array], layer: int, cfg: TinyConfig,
                         use_rope: bool = True) -> tuple[Array, Array]:
    """Project to Q, K, V; attend causally in every head; concatenate; project back."""
    t = x.shape[0]
    q = split_heads(x @ p[f"l{layer}.wq"], cfg.n_heads)
    k = split_heads(x @ p[f"l{layer}.wk"], cfg.n_heads)
    v = split_heads(x @ p[f"l{layer}.wv"], cfg.n_heads)
    if use_rope:
        pos = np.arange(t, dtype=float)
        q, k = rope(q, pos), rope(k, pos)
    out, weights = scaled_dot_product_attention(q, k, v, causal_mask(t))
    return merge_heads(out) @ p[f"l{layer}.wo"], weights


def feed_forward(x: Array, p: dict[str, Array], layer: int) -> Array:
    """Position-wise MLP: expand to d_ff, apply GELU, project back to d_model."""
    return gelu(x @ p[f"l{layer}.w1"] + p[f"l{layer}.b1"]) @ p[f"l{layer}.w2"] + p[f"l{layer}.b2"]


def decoder_block(x: Array, p: dict[str, Array], layer: int, cfg: TinyConfig) -> tuple[Array, Array]:
    """Pre-norm block: x + Attention(LN(x)), then x + FFN(LN(x)). The '+' are residual connections."""
    attn_out, weights = multi_head_attention(layer_norm(x, p[f"l{layer}.ln1.g"], p[f"l{layer}.ln1.b"]),
                                             p, layer, cfg)
    x = x + attn_out
    x = x + feed_forward(layer_norm(x, p[f"l{layer}.ln2.g"], p[f"l{layer}.ln2.b"]), p, layer)
    return x, weights


def forward(ids: Array, p: dict[str, Array], cfg: TinyConfig) -> tuple[Array, list[Array]]:
    """Token ids (T,) -> next-token logits (T, V), plus every layer's attention weights (H, T, T)."""
    if len(ids) > cfg.context_length:
        raise ValueError(f"sequence of {len(ids)} tokens exceeds context length {cfg.context_length}")
    x = p["tok_emb"][ids]  # (T, D)
    all_weights = []
    for layer in range(cfg.n_layers):
        x, w = decoder_block(x, p, layer, cfg)
        all_weights.append(w)
    x = layer_norm(x, p["ln_f.g"], p["ln_f.b"])
    return x @ p["tok_emb"].T, all_weights


def cross_entropy(logits: Array, targets: Array) -> float:
    """Mean negative log-probability of the correct next token (in nats)."""
    z = logits - logits.max(axis=-1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return float(-log_probs[np.arange(len(targets)), targets].mean())
