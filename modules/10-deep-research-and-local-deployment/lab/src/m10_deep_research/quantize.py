"""Symmetric round-to-nearest quantisation with numpy: per-tensor, per-channel, per-group.

The weights are seeded random numbers shaped like a linear layer, with uneven channel
scales and a few large outliers, because real weight matrices have both. Nothing here
is a measurement of a real model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Granularity = Literal["tensor", "channel", "group"]


def make_weights(
    rows: int = 256, cols: int = 1024, seed: int = 0, outlier_share: float = 0.001
) -> np.ndarray:
    """A synthetic weight matrix: rows have different scales and a few weights are outliers."""
    rng = np.random.default_rng(seed)
    row_scale = rng.lognormal(mean=0.0, sigma=0.5, size=(rows, 1))
    w = rng.normal(0.0, 0.02, size=(rows, cols)) * row_scale
    mask = rng.random((rows, cols)) < outlier_share
    w[mask] *= 10.0
    return w.astype(np.float32)


def quantize(
    w: np.ndarray, bits: int, granularity: Granularity, group_size: int = 128
) -> tuple[np.ndarray, int]:
    """Quantise then dequantise ``w``. Return (reconstructed weights, number of scales stored).

    Symmetric: code = round(w / scale) clipped to [-qmax, qmax], qmax = 2**(bits-1) - 1, and
    scale = max|w| / qmax over the tensor, each row (output channel) or each group of a row.
    """
    qmax = 2 ** (bits - 1) - 1
    rows, cols = w.shape
    if granularity == "tensor":
        blocks = w.reshape(1, -1)
    elif granularity == "channel":
        blocks = w
    else:
        if cols % group_size:
            raise ValueError(f"columns ({cols}) must be a multiple of group_size ({group_size})")
        blocks = w.reshape(-1, group_size)
    scale = np.abs(blocks).max(axis=1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    codes = np.clip(np.round(blocks / scale), -qmax, qmax)
    return (codes * scale).reshape(rows, cols).astype(np.float32), scale.size


@dataclass(frozen=True)
class QuantResult:
    name: str
    bits: int
    bits_per_weight: float  # including the 16-bit scales
    weight_error: float  # ||W - W_hat|| / ||W||
    output_error: float  # ||XW^T - XW_hat^T|| / ||XW^T|| on random inputs
    zero_share: float  # weights that became exactly 0 (information lost)


SCHEMES: list[tuple[str, int, Granularity, int]] = [
    ("int8 per-tensor", 8, "tensor", 0),
    ("int8 per-channel", 8, "channel", 0),
    ("int4 per-tensor", 4, "tensor", 0),
    ("int4 per-channel", 4, "channel", 0),
    ("int4 group-128", 4, "group", 128),
    ("int4 group-32", 4, "group", 32),
]


def compare(seed: int = 0) -> list[QuantResult]:
    """Run every scheme on the same seeded weights and inputs."""
    w = make_weights(seed=seed)
    x = np.random.default_rng(seed + 1).normal(size=(64, w.shape[1])).astype(np.float32)
    y = x @ w.T
    results = []
    for name, bits, gran, group in SCHEMES:
        w_hat, n_scales = quantize(w, bits, gran, group or 128)
        bpw = bits + 16 * n_scales / w.size
        werr = float(np.linalg.norm(w - w_hat) / np.linalg.norm(w))
        oerr = float(np.linalg.norm(y - x @ w_hat.T) / np.linalg.norm(y))
        zeros = float(np.mean((w_hat == 0) & (w != 0)))
        results.append(
            QuantResult(name, bits, round(bpw, 3), round(werr, 4), round(oerr, 4), round(zeros, 4))
        )
    return results
