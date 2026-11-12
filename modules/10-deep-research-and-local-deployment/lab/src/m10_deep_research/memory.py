"""How much memory does a model need? Weights + KV cache + overhead.

All sizes are in GB = 10**9 bytes. The model shapes used in the lessons are
illustrative examples, not the configuration of any specific released model.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1e9


def bits_per_weight(bits: int, group_size: int | None = None, scale_bits: int = 16) -> float:
    """Effective storage per weight: the integer code plus one shared scale per group."""
    return bits + (scale_bits / group_size if group_size else 0.0)


def weight_bytes(params: float, bits: int, group_size: int | None = None) -> float:
    return params * bits_per_weight(bits, group_size) / 8


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, kv_bits: int = 16) -> float:
    """Each layer stores one key and one value vector per KV head for every token."""
    return 2 * layers * kv_heads * head_dim * kv_bits / 8


def kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, context: int, batch: int = 1,
                   kv_bits: int = 16) -> float:
    return kv_bytes_per_token(layers, kv_heads, head_dim, kv_bits) * context * batch


@dataclass(frozen=True)
class MemoryEstimate:
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return round(self.weights_gb + self.kv_cache_gb + self.overhead_gb, 2)


def estimate(
    params: float,
    bits: int,
    context: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    *,
    batch: int = 1,
    kv_bits: int = 16,
    group_size: int | None = None,
    overhead: float = 0.10,
) -> MemoryEstimate:
    """Memory to serve a model: weights, KV cache for ``batch`` x ``context`` tokens, plus overhead.

    ``overhead`` is a fraction of (weights + KV cache) for activations, buffers and the
    runtime itself. 10% is a planning assumption, not a measurement: check yours.
    """
    w = weight_bytes(params, bits, group_size)
    kv = kv_cache_bytes(layers, kv_heads, head_dim, context, batch, kv_bits)
    return MemoryEstimate(round(w / GB, 2), round(kv / GB, 2), round(overhead * (w + kv) / GB, 2))


def decode_ceiling(bandwidth_gb_s: float, weights_gb: float) -> float:
    """Upper bound on tokens per second for one user: each new token reads every weight once.

    Real speed is lower (KV-cache reads, compute, overheads). Use it to compare options, not to promise.
    """
    return round(bandwidth_gb_s / weights_gb, 1)


@dataclass(frozen=True)
class FitRow:
    params: float
    bits: int
    total_gb: float
    fits: bool


def sweep(
    ram_gb: float,
    *,
    usable: float = 0.75,
    sizes: tuple[float, ...] = (1e9, 3e9, 8e9, 14e9, 32e9, 70e9),
    bit_widths: tuple[int, ...] = (16, 8, 4),
    context: int = 8192,
    layers: int = 32,
    kv_heads: int = 8,
    head_dim: int = 128,
    group_size: int = 32,
) -> list[FitRow]:
    """Which (size, bits) pairs fit in ``usable`` x ``ram_gb``? The OS and your apps need the rest.

    Simplification: every size uses the same example shape for the KV cache. Larger models
    usually have more layers, so their real KV cache is bigger than this estimate.
    """
    rows = []
    for p in sizes:
        for b in bit_widths:
            e = estimate(p, b, context, layers, kv_heads, head_dim, group_size=group_size if b < 16 else None)
            rows.append(FitRow(p, b, e.total_gb, e.total_gb <= usable * ram_gb))
    return rows
