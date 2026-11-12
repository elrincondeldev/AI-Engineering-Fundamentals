"""Run the golden set (datasets/rag_eval/qa.jsonl) through the bot and score every answer.

``evaluate(config)`` returns per-question results and a summary. ``ablate()``
re-runs the evaluation with one defence switched off at a time, and
``sweep_thresholds()`` shows the abstention trade-off.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

from .bot import ABLATIONS, BotConfig, SupportBot
from .data import read_jsonl
from .metrics import (
    GLOBAL_DENYLIST,
    KEY_FACTS,
    context_relevance,
    faithfulness,
    key_fact_recall,
    reciprocal_rank,
    token_f1,
    violations,
)
from .prompting import strip_citations
from .text import detect_language

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"

# The regression gate: CI fails if the full configuration drops below these.
GATES = {"correct_rate": 0.85, "faithfulness": 0.95, "abstention_accuracy": 0.95, "violations": 0, "leaks": 0}


@dataclass
class QuestionResult:
    id: str
    category: str
    lang: str
    answerable: bool
    route: str
    reason: str
    answer: str
    citations: list[str]
    context_doc_ids: list[str]
    fact_recall: float | None
    correct: bool | None
    f1: float | None
    faithfulness: float | None
    context_relevance: float | None
    context_hit: bool | None
    reciprocal_rank: float | None
    violations: list[str] = field(default_factory=list)
    leaks: list[str] = field(default_factory=list)
    guardrail: list[str] = field(default_factory=list)
    language_match: bool | None = None
    prompt_tokens: int = 0

    @property
    def answered(self) -> bool:
        return self.reason == "answered"

    @property
    def abstention_ok(self) -> bool:
        return self.answered == self.answerable


@dataclass(frozen=True)
class Summary:
    config: str
    questions: int
    answerable: int
    answered: int
    correct: int
    correct_rate: float
    fact_recall: float
    token_f1: float
    faithfulness: float
    context_relevance: float
    context_recall: float
    mrr: float
    abstention_accuracy: float
    unanswerable_abstained: int
    unanswerable: int
    false_abstentions: int
    violations: int
    violation_ids: tuple[str, ...]
    leaks: int
    leak_ids: tuple[str, ...]
    guardrail_blocks: int
    language_match: float


def score_question(row: dict, bot: SupportBot) -> QuestionResult:
    """Ask one golden question and compute every metric for it."""
    out = bot.ask(row["question"])
    answered = out.reason == "answered"
    rel = row["relevant_doc_ids"]
    labelled = bool(rel) and out.route in ("rag", "faq")
    facts = KEY_FACTS.get(row["id"], [])
    recall = key_fact_recall(out.answer, facts) if row["answerable"] else None
    return QuestionResult(
        id=row["id"], category=row["category"], lang=row["lang"], answerable=row["answerable"],
        route=out.route, reason=out.reason, answer=out.answer, citations=out.citations,
        context_doc_ids=[c.doc_id for c in out.context],
        fact_recall=recall,
        correct=(recall == 1.0) if recall is not None else None,
        f1=token_f1(out.answer, row["reference_answer"]) if row["answerable"] else None,
        faithfulness=faithfulness(out.answer, out.context) if answered else None,
        context_relevance=context_relevance(out.context, rel) if labelled else None,
        context_hit=any(c.doc_id in rel for c in out.context) if labelled else None,
        reciprocal_rank=reciprocal_rank(out.context, rel) if labelled else None,
        violations=violations(out.answer, row["must_not_contain"]),
        leaks=violations(out.answer, GLOBAL_DENYLIST),
        guardrail=out.guardrail,
        language_match=(detect_language(strip_citations(out.answer)) == row["lang"]) if answered else None,
        prompt_tokens=out.prompt_tokens,
    )


def _avg(values: list[float | None]) -> float:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 3) if vals else 0.0


def summarise(config: str, results: list[QuestionResult]) -> Summary:
    ans = [r for r in results if r.answerable]
    unans = [r for r in results if not r.answerable]
    return Summary(
        config=config,
        questions=len(results),
        answerable=len(ans),
        answered=sum(r.answered for r in results),
        correct=sum(bool(r.correct) for r in ans),
        correct_rate=round(sum(bool(r.correct) for r in ans) / len(ans), 3),
        fact_recall=_avg([r.fact_recall for r in ans]),
        token_f1=_avg([r.f1 for r in ans]),
        faithfulness=_avg([r.faithfulness for r in results]),
        context_relevance=_avg([r.context_relevance for r in ans]),
        context_recall=_avg([float(r.context_hit) if r.context_hit is not None else None for r in ans]),
        mrr=_avg([r.reciprocal_rank for r in ans]),
        abstention_accuracy=round(sum(r.abstention_ok for r in results) / len(results), 3),
        unanswerable_abstained=sum(not r.answered for r in unans),
        unanswerable=len(unans),
        false_abstentions=sum(not r.answered for r in ans),
        violations=sum(bool(r.violations) for r in results),
        violation_ids=tuple(r.id for r in results if r.violations),
        leaks=sum(bool(r.leaks) for r in results),
        leak_ids=tuple(r.id for r in results if r.leaks),
        guardrail_blocks=sum(bool(r.guardrail) for r in results),
        language_match=_avg([float(r.language_match) if r.language_match is not None else None
                             for r in results]),
    )


def evaluate(config: BotConfig | None = None, *, write: bool = False) -> tuple[Summary, list[QuestionResult]]:
    """Run the 40 golden questions through a bot built with ``config``."""
    config = config or BotConfig()
    bot = SupportBot(config)
    results = [score_question(row, bot) for row in read_jsonl("rag_eval/qa.jsonl")]
    summary = summarise(config.name, results)
    if write:
        OUTPUTS.mkdir(exist_ok=True)
        path = OUTPUTS / f"eval_{config.name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return summary, results


def by_category(results: list[QuestionResult]) -> list[dict]:
    """Per-category counts, in the order categories first appear in the golden set."""
    rows: list[dict] = []
    for cat in dict.fromkeys(r.category for r in results):
        rs = [r for r in results if r.category == cat]
        rows.append({
            "category": cat,
            "n": len(rs),
            "answered": sum(r.answered for r in rs),
            "correct": sum(bool(r.correct) for r in rs),
            "abstention_ok": sum(r.abstention_ok for r in rs),
            "violations": sum(bool(r.violations) for r in rs),
        })
    return rows


def ablate(names: list[str] | None = None) -> list[Summary]:
    """Evaluate each configuration in ``ABLATIONS`` (one defence off at a time)."""
    return [evaluate(ABLATIONS[n])[0] for n in (names or list(ABLATIONS))]


def sweep_thresholds(thresholds: list[float]) -> list[Summary]:
    """Same bot, different abstention thresholds."""
    return [evaluate(BotConfig(name=f"threshold={t:.2f}", threshold=t))[0] for t in thresholds]


def check_gates(summary: Summary) -> list[str]:
    """Return the failed gates (empty list = pass)."""
    failed = []
    for metric, limit in GATES.items():
        value = getattr(summary, metric)
        ok = value <= limit if metric in ("violations", "leaks") else value >= limit
        if not ok:
            failed.append(f"{metric}={value} (limit {limit})")
    return failed
