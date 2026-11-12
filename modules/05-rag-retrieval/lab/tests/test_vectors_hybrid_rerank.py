"""Hashing embeddings, the exact vector index, reciprocal rank fusion and the reranker."""

from __future__ import annotations

import math

import numpy as np

from m05_retrieval.hybrid import rrf_fuse
from m05_retrieval.rerank import coverage_score, rerank
from m05_retrieval.vectors import HashingEmbedder, VectorIndex, normalize


def test_normalize_gives_unit_rows() -> None:
    m = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(m[0], [0.6, 0.8]) and np.allclose(m[1], [0.0, 0.0])


def test_vector_index_finds_the_closest_text() -> None:
    texts = ["reset the hearth hub", "glow bulb brightness", "shipping to chile"]
    index = VectorIndex(texts, HashingEmbedder())
    assert index.search("how do I reset my hub", k=1)[0][0] == 0
    assert index.nbytes == 3 * 256 * 4


def test_hashing_embedder_has_no_idea_of_synonyms_or_languages() -> None:
    emb = HashingEmbedder()
    a, b, c, d = emb.embed(["price", "cost", "return window", "plazo de devolución"])
    assert abs(float(a @ b)) < 0.2  # synonyms look unrelated
    assert abs(float(c @ d)) < 0.2  # so does the same idea in Spanish


def test_rrf_uses_ranks_only() -> None:
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    assert [item for item, _ in fused] == [1, 3, 2, 4]
    assert math.isclose(fused[0][1], 1 / 61 + 1 / 62)


def test_coverage_matches_word_prefixes() -> None:
    assert coverage_score("How much does the Watcher S1 weigh?", "Watcher S1 | Weight (g): 34") == 0.75
    assert coverage_score("the of and", "anything") == 0.0
    order = rerank("watcher weigh", [(7, "watcher only", 9.0), (8, "watcher weight 34", 1.0)])
    assert [cid for cid, _ in order] == [8, 7]
