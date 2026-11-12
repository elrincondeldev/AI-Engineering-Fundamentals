"""Retrieval metrics and the end-to-end pipeline."""

from __future__ import annotations

from m05_retrieval.chunking import Chunk
from m05_retrieval.evaluate import doc_ranking, load_questions, recall_at_k, reciprocal_rank
from m05_retrieval.retrieval import Config, Hit, build_pipeline


def _hit(doc: str, n: int) -> Hit:
    return Hit(Chunk(f"{doc}#{n}", doc, "text"), 1.0)


def test_doc_ranking_collapses_chunks() -> None:
    hits = [_hit("a", 0), _hit("a", 1), _hit("b", 0), _hit("c", 0), _hit("b", 1)]
    assert doc_ranking(hits) == ["a", "b", "c"]


def test_recall_and_reciprocal_rank() -> None:
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, ["c"], 2) == 0.0 and recall_at_k(ranked, ["c"], 3) == 1.0
    assert reciprocal_rank(ranked, ["b", "c"]) == 0.5
    assert reciprocal_rank(ranked, ["z"]) == 0.0


def test_question_set() -> None:
    assert len(load_questions()) == 36
    assert len(load_questions(answerable_only=False)) == 40


def test_filtered_pipeline_never_returns_blocked_documents() -> None:
    pipe = build_pipeline(Config("hybrid", "table"))
    for q in ("goodwill credit code", "restocking fee for opened items", "account and billing faq export"):
        found = {h.chunk.doc_id for h in pipe.retriever.search(q, 20)}
        assert found.isdisjoint({"kb-010", "kb-041", "kb-027"})


def test_unfiltered_pipeline_surfaces_the_old_policy_first() -> None:
    pipe = build_pipeline(Config("bm25", "heading", filter_meta=False))
    top = pipe.retriever.search("What is the return window for Hearthwise products?", 1)[0]
    assert top.chunk.doc_id == "kb-010"
