"""Evaluation harness: accuracy on a held-out set, plus the slices that matter to Alex and Dani."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass

from .icl import ToyInContextLearner
from .prompts import ExampleSelector, PromptConfig, build_messages
from .providers import FakeProvider, LLMProvider, get_provider
from .structured import classify
from .tickets import INTENTS, Ticket


def make_provider(learner: ToyInContextLearner | None = None) -> LLMProvider:
    """``LLM_PROVIDER=fake`` (default) gives the toy in-context learner; otherwise a real model."""
    chosen = (os.environ.get("LLM_PROVIDER") or "fake").strip().lower()
    if chosen == "fake":
        return FakeProvider(responder=learner or ToyInContextLearner(), model="toy-icl")
    return get_provider(chosen)


@dataclass(frozen=True)
class EvalResult:
    """Accuracy overall and on the slices we care about, plus cost and format statistics."""

    name: str
    n: int
    accuracy: float
    accuracy_camera: float
    accuracy_spanish: float
    prompt_tokens: float  # mean per ticket, including retries
    completion_tokens: float
    valid_first_try: float
    unparseable: int

    def row(self) -> str:
        return (f"{self.name:<34} {self.accuracy:>6.3f} {self.accuracy_camera:>7.3f} {self.accuracy_spanish:>7.3f}"
                f" {self.prompt_tokens:>8.1f}")


HEADER = f"{'prompt':<34} {'acc':>6} {'camera':>7} {'spanish':>7} {'tokens':>8}"


def _share(hits: list[bool]) -> float:
    return sum(hits) / len(hits) if hits else 0.0


def evaluate_prompt(
    name: str,
    test: list[Ticket],
    pool: list[Ticket],
    cfg: PromptConfig,
    provider: LLMProvider | None = None,
    max_retries: int = 1,
) -> EvalResult:
    """Route every test ticket with the prompt described by ``cfg`` and score the answers."""
    provider = provider or make_provider()
    selector = ExampleSelector(pool, cfg) if cfg.k > 0 else None
    correct: list[bool] = []
    camera: list[bool] = []
    spanish: list[bool] = []
    prompt_tok = completion_tok = 0
    first_ok = unparseable = 0
    for ticket in test:
        result = classify(provider, build_messages(ticket, cfg, selector), INTENTS,
                          json_mode=cfg.output == "json", max_retries=max_retries)
        hit = result.label == ticket.intent
        correct.append(hit)
        if ticket.product == "Lookout C2":
            camera.append(hit)
        if ticket.language == "es":
            spanish.append(hit)
        prompt_tok += result.prompt_tokens
        completion_tok += result.completion_tokens
        first_ok += result.valid_first_try
        unparseable += result.label is None
    n = len(test)
    return EvalResult(
        name=name,
        n=n,
        accuracy=round(_share(correct), 3),
        accuracy_camera=round(_share(camera), 3),
        accuracy_spanish=round(_share(spanish), 3),
        prompt_tokens=round(prompt_tok / n, 1),
        completion_tokens=round(completion_tok / n, 1),
        valid_first_try=round(first_ok / n, 3),
        unparseable=unparseable,
    )


def majority_baseline(train: list[Ticket], test: list[Ticket]) -> float:
    """Always predict the most common training label."""
    top = Counter(t.intent for t in train).most_common(1)[0][0]
    return round(sum(t.intent == top for t in test) / len(test), 3)


def label_shares(tickets: list[Ticket]) -> dict[str, float]:
    counts = Counter(t.intent for t in tickets)
    return {label: round(counts[label] / len(tickets), 3) for label in INTENTS}


@dataclass(frozen=True)
class ColumnAudit:
    """How well one column alone predicts the label on the test set."""

    column: str
    distinct_values: int
    accuracy: float
    known_at_prediction_time: bool


KNOWN_AT_PREDICTION_TIME = {
    "channel": True,
    "language": True,
    "product": True,
    "customer_email": True,
    "resolution_code": False,  # written by the agent after the ticket is solved
}


def single_column_accuracy(train: list[Ticket], test: list[Ticket], column: str) -> float:
    """Learn 'value -> most common label' on train; unseen values get the overall majority."""
    by_value: dict[str, Counter[str]] = defaultdict(Counter)
    for t in train:
        by_value[getattr(t, column)][t.intent] += 1
    fallback = Counter(t.intent for t in train).most_common(1)[0][0]
    rule = {value: counts.most_common(1)[0][0] for value, counts in by_value.items()}
    hits = sum(rule.get(getattr(t, column), fallback) == t.intent for t in test)
    return round(hits / len(test), 3)


def audit_columns(train: list[Ticket], test: list[Ticket]) -> list[ColumnAudit]:
    """One row per metadata column: a single column that predicts the label perfectly is a red flag."""
    return [
        ColumnAudit(col, len({getattr(t, col) for t in train}), single_column_accuracy(train, test, col), known)
        for col, known in KNOWN_AT_PREDICTION_TIME.items()
    ]
