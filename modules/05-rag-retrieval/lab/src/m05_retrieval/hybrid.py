"""Reciprocal rank fusion (RRF): merge several rankings using ranks only, never raw scores."""

from __future__ import annotations

from collections import defaultdict


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked lists of item ids. Each list adds ``1 / (k + rank)`` (rank starts at 1) per item.

    Returns (item, fused score) pairs, best first. Ties keep the order in which items were first seen.
    """
    fused: dict[int, float] = defaultdict(float)
    first_seen: dict[int, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
