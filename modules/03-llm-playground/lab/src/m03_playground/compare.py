"""Compare mode: the same prompt across models and temperatures, several samples each.

One sample tells you almost nothing about a sampled model. Compare mode runs
``samples`` seeds per setting and summarises them side by side: how many
different answers, how long, how slow, how expensive and how many failed an
output check.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .metrics import EXAMPLE_PRICES
from .playground import Playground, Settings


@dataclass
class CompareRow:
    model: str
    temperature: float
    samples: int
    distinct: int  # how many different answers across the seeds
    mean_completion_tokens: float
    mean_ttft_ms: float
    mean_latency_ms: float
    cost_per_1k_eur: float | None  # cost of 1,000 such requests at example prices
    flagged: int  # answers with at least one output-check flag
    answers: dict[str, int] = field(default_factory=dict)  # answer text -> count


def sweep(
    prompt: str,
    *,
    models: Sequence[str] = ("fake-small", "fake-large"),
    temperatures: Sequence[float] = (0.0, 0.7, 1.5),
    samples: int = 10,
    max_tokens: int = 256,
    provider: str = "fake",
) -> list[CompareRow]:
    rows: list[CompareRow] = []
    for model in models:
        for temp in temperatures:
            answers: dict[str, int] = {}
            tokens, ttfts, lats, costs, flagged = [], [], [], [], 0
            for seed in range(samples):
                pg = Playground(
                    Settings(
                        provider=provider, model=model, temperature=temp, seed=seed, max_tokens=max_tokens
                    ),
                    session_id=f"compare-{seed}",
                )
                r = pg.ask(prompt)
                answers[r.answer] = answers.get(r.answer, 0) + 1
                tokens.append(r.completion.completion_tokens)
                ttfts.append(r.ttft_ms or 0.0)
                lats.append(r.completion.latency_ms)
                costs.append(r.cost_eur)
                flagged += bool(r.output_flags)
            known = [c for c in costs if c is not None]
            rows.append(
                CompareRow(
                    model=model,
                    temperature=temp,
                    samples=samples,
                    distinct=len(answers),
                    mean_completion_tokens=sum(tokens) / samples,
                    mean_ttft_ms=sum(ttfts) / samples,
                    mean_latency_ms=sum(lats) / samples,
                    cost_per_1k_eur=1000 * sum(known) / len(known) if len(known) == samples else None,
                    flagged=flagged,
                    answers=answers,
                )
            )
    return rows


def example_prices_note(models: Sequence[str]) -> str:
    parts = [
        f"{m} €{p.input_per_mtok:.2f}/€{p.output_per_mtok:.2f}"
        for m in models
        if (p := EXAMPLE_PRICES.get(m))
    ]
    return "example prices per million input/output tokens: " + ", ".join(parts) if parts else ""
