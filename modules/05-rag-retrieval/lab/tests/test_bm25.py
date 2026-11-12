"""Inverted index and BM25 behave as the lesson describes."""

from __future__ import annotations

import math

from m05_retrieval.bm25 import BM25, InvertedIndex, idf

DOCS = [
    "the hub connects devices",
    "the bulb is bright",
    "reset the hub hub hub",
    "warranty for every device",
]


def test_inverted_index_postings() -> None:
    index = InvertedIndex.build(DOCS)
    assert index.postings["hub"] == {0: 1, 2: 3}
    assert index.lengths == [4, 4, 5, 4]
    assert index.boolean_and(["the", "hub"]) == {0, 2}
    assert index.n_postings == sum(len(set(d.split())) for d in DOCS)


def test_idf_rewards_rare_terms() -> None:
    assert idf(4, 1) > idf(4, 3) > 0
    assert math.isclose(idf(4, 1), math.log(1 + 3.5 / 1.5))


def test_term_frequency_saturates() -> None:
    bm = BM25(DOCS)
    one, three, thirty = (bm.term_score(tf, 4, 1.0) for tf in (1, 3, 30))
    assert one < three < thirty < bm.k1 + 1  # more is better, but never more than k1 + 1 times the IDF


def test_longer_chunks_are_penalised() -> None:
    bm = BM25(DOCS)
    assert bm.term_score(1, 2, 1.0) > bm.term_score(1, 20, 1.0)
    flat = BM25(DOCS, b=0.0)
    assert flat.term_score(1, 2, 1.0) == flat.term_score(1, 20, 1.0)


def test_scores_rank_the_right_chunk_first() -> None:
    scores = BM25(DOCS).scores("reset hub")
    assert max(range(len(DOCS)), key=scores.__getitem__) == 2
    assert scores[1] == 0.0


def test_accent_folding_is_optional() -> None:
    texts = ["Enviamos a México", "Shipping to Spain"]
    assert BM25(texts).scores("Mexico")[0] == 0.0
    assert BM25(texts, fold=True).scores("Mexico")[0] > 0.0
