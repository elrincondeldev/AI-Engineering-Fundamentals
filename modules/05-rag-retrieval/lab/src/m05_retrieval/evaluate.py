"""Retrieval metrics on ``datasets/rag_eval/qa.jsonl``: document recall@k and MRR, per category."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .data import read_jsonl
from .retrieval import Hit, Retriever


def load_questions(*, answerable_only: bool = True) -> list[dict[str, Any]]:
    """The evaluation questions. By default only those with at least one relevant document."""
    rows = read_jsonl("rag_eval/qa.jsonl")
    return [q for q in rows if q["relevant_doc_ids"]] if answerable_only else rows


def doc_ranking(hits: list[Hit]) -> list[str]:
    """Collapse a chunk ranking into a document ranking (first appearance wins)."""
    docs: list[str] = []
    for h in hits:
        if h.chunk.doc_id not in docs:
            docs.append(h.chunk.doc_id)
    return docs


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    """1.0 if any relevant document is in the top k, else 0.0 ('hit rate' style recall)."""
    return float(any(d in relevant for d in ranked[:k]))


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    """1 / position of the first relevant document; 0 if none is retrieved."""
    for pos, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / pos
    return 0.0


@dataclass
class EvalReport:
    name: str
    n: int
    recall_at_1: float
    recall_at_5: float
    mrr: float
    context_at_5: float  # a relevant document among the first 5 *chunks* (what a prompt would hold)
    by_category: dict[str, float] = field(default_factory=dict)  # recall@5 per category
    blocked_in_top5: int = 0  # questions (all 40) whose top 5 holds a superseded or internal document
    misses: list[str] = field(default_factory=list)  # question ids with recall@5 = 0


def evaluate(retriever: Retriever, name: str = "", depth: int = 50) -> EvalReport:
    """Score the document ranking of every answerable question; check exposure on all questions."""
    r1, r5, rr, ctx, blocked = [], [], [], [], 0
    cats: dict[str, list[float]] = defaultdict(list)
    misses = []
    for q in load_questions(answerable_only=False):
        hits = retriever.search(q["question"], depth)
        ranked = doc_ranking(hits)
        top5 = {h.chunk.doc_id: h.chunk.meta for h in hits if h.chunk.doc_id in ranked[:5]}
        if any(m.get("status") != "current" or m.get("audience") != "public" for m in top5.values()):
            blocked += 1
        rel = q["relevant_doc_ids"]
        if not rel:
            continue
        r1.append(recall_at_k(ranked, rel, 1))
        r5.append(recall_at_k(ranked, rel, 5))
        rr.append(reciprocal_rank(ranked, rel))
        ctx.append(float(any(h.chunk.doc_id in rel for h in hits[:5])))
        cats[q["category"]].append(r5[-1])
        if r5[-1] == 0:
            misses.append(q["id"])
    n = len(r5)
    return EvalReport(
        name=name,
        n=n,
        recall_at_1=round(sum(r1) / n, 3),
        recall_at_5=round(sum(r5) / n, 3),
        mrr=round(sum(rr) / n, 3),
        context_at_5=round(sum(ctx) / n, 3),
        by_category={c: round(sum(v) / len(v), 2) for c, v in sorted(cats.items())},
        blocked_in_top5=blocked,
        misses=misses,
    )
