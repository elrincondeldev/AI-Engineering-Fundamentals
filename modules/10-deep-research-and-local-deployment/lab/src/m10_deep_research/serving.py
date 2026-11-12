"""Back-of-the-envelope serving maths: batching, KV-cache paging, speculative decoding, cost.

The batching simulator uses a made-up step-time model (milliseconds per decode step
and per prompt token). It shows *why* schedulers behave differently; its numbers are
simulated, not benchmarks of any engine or hardware.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    id: int
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int


def make_requests(n: int = 64, seed: int = 0, mean_gap_ms: float = 1500.0) -> list[Request]:
    """Seeded workload: random arrivals, prompts of 100-1,000 tokens, answers of 16-512 tokens."""
    rng = random.Random(seed)
    t, out = 0.0, []
    for i in range(n):
        t += rng.expovariate(1 / mean_gap_ms)
        out.append(Request(i, round(t, 1), rng.randint(100, 1000), rng.randint(16, 512)))
    return out


@dataclass(frozen=True)
class StepModel:
    """Illustrative costs: a decode step costs base + per_seq x batch; prefill costs per prompt token."""

    decode_base_ms: float = 25.0
    decode_per_seq_ms: float = 1.0
    prefill_per_token_ms: float = 0.1

    def decode(self, batch: int) -> float:
        return self.decode_base_ms + self.decode_per_seq_ms * batch

    def prefill(self, tokens: int) -> float:
        return self.prefill_per_token_ms * tokens


@dataclass(frozen=True)
class SimResult:
    scheduler: str
    ttft_p50_ms: float
    ttft_p95_ms: float
    tpot_mean_ms: float
    throughput_tok_s: float


def _summarise(name: str, reqs: list[Request], first: dict[int, float], done: dict[int, float]) -> SimResult:
    ttft = sorted(first[r.id] - r.arrival_ms for r in reqs)
    tpot = [(done[r.id] - first[r.id]) / (r.output_tokens - 1) for r in reqs]
    span_s = (max(done.values()) - min(r.arrival_ms for r in reqs)) / 1000
    p95 = ttft[min(len(ttft) - 1, math.ceil(0.95 * len(ttft)) - 1)]
    return SimResult(name, round(statistics.median(ttft), 1), round(p95, 1), round(statistics.mean(tpot), 1),
                     round(sum(r.output_tokens for r in reqs) / span_s, 1))


def simulate_static(reqs: list[Request], max_batch: int, model: StepModel = StepModel()) -> SimResult:
    """Static batching: take the requests waiting, run them until the *longest* one finishes."""
    pending = deque(sorted(reqs, key=lambda r: r.arrival_ms))
    clock, first, done = 0.0, {}, {}
    while pending:
        clock = max(clock, pending[0].arrival_ms)
        batch = []
        while pending and pending[0].arrival_ms <= clock and len(batch) < max_batch:
            batch.append(pending.popleft())
        clock += model.prefill(sum(r.prompt_tokens for r in batch))
        for r in batch:
            first[r.id] = clock
        for step in range(1, max(r.output_tokens for r in batch)):
            clock += model.decode(len(batch))  # finished sequences still hold their slot
            for r in batch:
                if r.output_tokens - 1 == step:
                    done[r.id] = clock
        for r in batch:
            done.setdefault(r.id, clock)
    return _summarise(f"static (batch {max_batch})", reqs, first, done)


def simulate_continuous(reqs: list[Request], max_batch: int, model: StepModel = StepModel()) -> SimResult:
    """Continuous batching: after every step, finished sequences leave and waiting ones join."""
    pending = deque(sorted(reqs, key=lambda r: r.arrival_ms))
    running: dict[int, int] = {}  # request id -> tokens still to generate
    clock, first, done = 0.0, {}, {}
    while pending or running:
        if not running and pending and pending[0].arrival_ms > clock:
            clock = pending[0].arrival_ms
        joining = []
        while pending and pending[0].arrival_ms <= clock and len(running) + len(joining) < max_batch:
            joining.append(pending.popleft())
        step = model.prefill(sum(r.prompt_tokens for r in joining))
        if running:
            step += model.decode(len(running))
        clock += step
        for rid in list(running):
            running[rid] -= 1
            if running[rid] == 0:
                done[rid] = clock
                del running[rid]
        for r in joining:
            first[r.id] = clock
            if r.output_tokens > 1:
                running[r.id] = r.output_tokens - 1
            else:
                done[r.id] = clock
    return _summarise(f"continuous (batch {max_batch})", reqs, first, done)


# --------------------------------------------------------------------------- #
# KV-cache allocation: contiguous reservation versus paging                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KVPlan:
    strategy: str
    tokens_reserved: int  # for all requests together
    tokens_used: int
    concurrent_fit: int  # requests that fit in the budget at the same time

    @property
    def waste(self) -> float:
        return round(1 - self.tokens_used / self.tokens_reserved, 3)


def kv_plans(reqs: list[Request], kv_budget_gb: float, bytes_per_token: float, max_seq_len: int = 8192,
             block: int = 16) -> list[KVPlan]:
    """Compare reserving ``max_seq_len`` per request with allocating ``block``-token pages on demand."""
    used = [r.prompt_tokens + r.output_tokens for r in reqs]
    budget_tokens = int(kv_budget_gb * 1e9 // bytes_per_token)
    plans = []
    for name, need in (("contiguous", [max_seq_len] * len(used)),
                       ("paged", [math.ceil(u / block) * block for u in used])):
        fit, total = 0, 0
        for n in need:
            if total + n > budget_tokens:
                break
            fit, total = fit + 1, total + n
        plans.append(KVPlan(name, sum(need), sum(used), fit))
    return plans


# --------------------------------------------------------------------------- #
# Speculative decoding (expected values from Leviathan et al., 2022)          #
# --------------------------------------------------------------------------- #


def speculative(alpha: float, k: int, cost_ratio: float) -> tuple[float, float]:
    """Return (tokens per target-model pass, expected wall-time speed-up).

    ``alpha``: chance the target model accepts a draft token. ``k``: draft tokens per round.
    ``cost_ratio``: time of one draft-model step / time of one target-model step.
    """
    tokens = (1 - alpha ** (k + 1)) / (1 - alpha) if alpha < 1 else k + 1.0
    return round(tokens, 3), round(tokens / (k * cost_ratio + 1), 3)


# --------------------------------------------------------------------------- #
# Cost and capacity (every price is an input: plug in your own)               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CostInputs:
    tokens_in_per_day: float
    tokens_out_per_day: float
    api_eur_per_m_in: float
    api_eur_per_m_out: float
    hardware_eur: float
    amortisation_months: int
    power_watts: float
    busy_hours_per_day: float
    eur_per_kwh: float


@dataclass(frozen=True)
class CostResult:
    api_eur_per_day: float
    local_eur_per_day: float
    local_hardware_eur_per_day: float
    local_energy_eur_per_day: float
    breakeven_multiplier: float  # how many times today's volume makes local cheaper


def daily_cost(c: CostInputs) -> CostResult:
    api = c.tokens_in_per_day / 1e6 * c.api_eur_per_m_in + c.tokens_out_per_day / 1e6 * c.api_eur_per_m_out
    hardware = c.hardware_eur / (c.amortisation_months * 30)
    energy = c.power_watts / 1000 * c.busy_hours_per_day * c.eur_per_kwh
    local = hardware + energy
    breakeven = local / api if api else math.inf
    return CostResult(
        round(api, 2), round(local, 2), round(hardware, 2), round(energy, 2), round(breakeven, 1)
    )


def servers_needed(peak_tokens_per_s: float, server_tokens_per_s: float, headroom: float = 0.3) -> int:
    """Replicas to serve the peak with ``headroom`` spare capacity (never plan to run at 100%)."""
    return math.ceil(peak_tokens_per_s * (1 + headroom) / server_tokens_per_s)
