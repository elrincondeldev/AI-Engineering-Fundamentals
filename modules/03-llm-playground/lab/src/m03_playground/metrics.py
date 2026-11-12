"""Latency and cost metrics that Sam (SRE) asked for.

Prices are always *parameters*: they change often and differ per contract.
``EXAMPLE_PRICES`` are invented numbers for the simulated models, not real prices.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Prices:
    """Euros per million tokens. Input (prompt) and output (completion) tokens are priced separately."""

    input_per_mtok: float
    output_per_mtok: float


# Example prices for the simulated models. NOT real prices of any provider.
EXAMPLE_PRICES: dict[str, Prices] = {
    "fake-small": Prices(input_per_mtok=0.10, output_per_mtok=0.40),
    "fake-large": Prices(input_per_mtok=1.00, output_per_mtok=4.00),
}
FREE = Prices(0.0, 0.0)


def request_cost(prompt_tokens: int, completion_tokens: int, prices: Prices) -> float:
    """Cost of one request in euros."""
    return (prompt_tokens * prices.input_per_mtok + completion_tokens * prices.output_per_mtok) / 1_000_000


def decode_tokens_per_second(n_tokens: int, ttft_ms: float, total_ms: float) -> float:
    """Generation speed after the first token: (n - 1) tokens over (total - TTFT)."""
    if n_tokens < 2 or total_ms <= ttft_ms:
        return 0.0
    return (n_tokens - 1) / ((total_ms - ttft_ms) / 1000)


def end_to_end_tokens_per_second(n_tokens: int, total_ms: float) -> float:
    """What the user experiences: all tokens over the whole request time (includes waiting)."""
    return n_tokens / (total_ms / 1000) if total_ms > 0 else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). Simple and exact on small samples."""
    if not values:
        raise ValueError("percentile of an empty list")
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class Workload:
    """A traffic assumption. Every field is a parameter you must justify, not a fact."""

    users: int
    requests_per_user_per_day: float
    working_days_per_month: int
    prompt_tokens: int
    completion_tokens: int

    @property
    def requests_per_month(self) -> int:
        return round(self.users * self.requests_per_user_per_day * self.working_days_per_month)


@dataclass(frozen=True)
class CostEstimate:
    per_request: float
    per_day: float
    per_month: float


def estimate_cost(workload: Workload, prices: Prices) -> CostEstimate:
    """Project a workload into per-request, per-day and per-month cost."""
    per_request = request_cost(workload.prompt_tokens, workload.completion_tokens, prices)
    per_day = per_request * workload.users * workload.requests_per_user_per_day
    return CostEstimate(per_request, per_day, per_request * workload.requests_per_month)
