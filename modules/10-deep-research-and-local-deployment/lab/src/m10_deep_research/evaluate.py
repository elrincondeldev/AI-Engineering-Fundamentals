"""Run the research loop on every question, grade the reports and compare ablations."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .corpus import Snapshot, load_labels, load_questions, load_snapshot
from .grade import Grade, grade_report
from .loop import ResearchConfig, ResearchResult, run_research
from .providers import LLMProvider
from .search import BM25Index

# Each ablation switches off one defence (or two, to show defence in depth).
ABLATIONS: dict[str, ResearchConfig] = {
    "full": ResearchConfig(),
    "no-credibility": ResearchConfig(use_credibility=False),
    "no-recency": ResearchConfig(use_recency=False),
    "no-sanitiser": ResearchConfig(use_sanitiser=False),
    "no-dedup": ResearchConfig(use_dedup=False),
    "no-credibility+no-sanitiser": ResearchConfig(use_credibility=False, use_sanitiser=False),
}


@dataclass
class Workspace:
    """Everything loaded once: snapshot, search index, questions and ground-truth labels."""

    snapshot: Snapshot
    index: BM25Index
    questions: dict[str, dict]
    labels: dict[str, dict]


def load_workspace() -> Workspace:
    snapshot = load_snapshot()
    return Workspace(snapshot, BM25Index(snapshot.searchable()), load_questions(), load_labels())


def run_and_grade(
    ws: Workspace, qid: str, config: ResearchConfig = ResearchConfig(), provider: LLMProvider | None = None
) -> tuple[ResearchResult, Grade]:
    result = run_research(ws.questions[qid], ws.snapshot, ws.index, provider, config)
    return result, grade_report(result.report, ws.questions[qid], ws.snapshot, ws.labels)


@dataclass(frozen=True)
class Row:
    """Totals over all questions for one configuration."""

    name: str
    facts_covered: int
    facts_total: int
    valid_cited_lines: int
    cited_lines: int
    low_credibility_citations: int
    trap_hits: int
    pii_leaks: int
    questions_with_enough_sources: int
    tokens_kept: int


def summarise(name: str, pairs: list[tuple[ResearchResult, Grade]]) -> Row:
    grades = [g for _, g in pairs]
    return Row(
        name=name,
        facts_covered=sum(g.facts_covered for g in grades),
        facts_total=sum(g.facts_total for g in grades),
        valid_cited_lines=sum(g.valid_cited_lines for g in grades),
        cited_lines=sum(g.cited_lines for g in grades),
        low_credibility_citations=sum(g.low_credibility_citations for g in grades),
        trap_hits=sum(len(g.trap_hits) for g in grades),
        pii_leaks=sum(g.pii_leaks for g in grades),
        questions_with_enough_sources=sum(g.enough_sources for g in grades),
        tokens_kept=sum(r.stats.tokens_kept for r, _ in pairs),
    )


def run_ablations(ws: Workspace, names: list[str] | None = None) -> list[Row]:
    rows = []
    for name in names or list(ABLATIONS):
        pairs = [run_and_grade(ws, qid, ABLATIONS[name]) for qid in sorted(ws.questions)]
        rows.append(summarise(name, pairs))
    return rows


def with_overrides(config: ResearchConfig, **overrides: object) -> ResearchConfig:
    """Return a copy of ``config`` with some fields changed (CLI flags use this)."""
    return replace(config, **{k: v for k, v in overrides.items() if v is not None})
