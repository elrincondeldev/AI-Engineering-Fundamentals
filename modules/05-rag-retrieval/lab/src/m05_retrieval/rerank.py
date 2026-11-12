"""A small lexical reranker: a second, slower look at the top candidates only.

Production systems use a cross-encoder model here (it reads the query and the
chunk together). This stand-in scores how much of the query a chunk covers,
matching word prefixes so that 'weigh' meets 'weight'. It is deterministic and
offline, which is enough to show where reranking sits in the pipeline.
"""

from __future__ import annotations

from .text import content_terms, tokenize


def _stem(word: str, size: int = 5) -> str:
    return word[:size]


def coverage_score(query: str, text: str) -> float:
    """Share of the query's content terms that appear in ``text`` (prefix match), in [0, 1]."""
    terms = content_terms(query)
    if not terms:
        return 0.0
    stems = {_stem(w) for w in tokenize(text)}
    return sum(_stem(t) in stems for t in terms) / len(terms)


def rerank(query: str, candidates: list[tuple[int, str, float]]) -> list[tuple[int, float]]:
    """Reorder (id, text, first-stage score) candidates by coverage, breaking ties by first-stage order.

    Returns (id, rerank score) pairs, best first.
    """
    scored = [(cid, coverage_score(query, text), pos) for pos, (cid, text, _) in enumerate(candidates)]
    scored.sort(key=lambda t: (-t[1], t[2]))
    return [(cid, score) for cid, score, _ in scored]
