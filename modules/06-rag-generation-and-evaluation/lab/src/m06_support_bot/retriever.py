"""A small BM25 retriever with metadata filters applied *before* ranking.

Filtering first is access control: a chunk that the customer may not see
(``audience: internal``) or that is no longer valid (``status: superseded``)
never reaches the ranking, the prompt or the logs.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .kb import Chunk
from .text import content_tokens


@dataclass(frozen=True)
class Filters:
    """Which chunks a request may see. ``None`` disables a filter (used in ablations)."""

    audiences: frozenset[str] | None = frozenset({"public"})
    statuses: frozenset[str] | None = frozenset({"current"})

    def allows(self, chunk: Chunk) -> bool:
        if self.audiences is not None and chunk.doc.audience not in self.audiences:
            return False
        return self.statuses is None or chunk.doc.status in self.statuses


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


class BM25Index:
    """Okapi BM25 over 'title + section + body' of each chunk."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [content_tokens(f"{c.doc.title} {c.section} {c.text}") for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.avg_len = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        self.n = len(self.docs)
        self._idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.max_idf = math.log(1 + (self.n + 0.5) / 0.5)  # a term that appears nowhere

    def idf(self, term: str) -> float:
        """Rarity of a (stemmed) term. Unknown terms get the maximum: they are very informative."""
        return self._idf.get(term, self.max_idf)

    def search(self, terms: list[str], *, k: int = 4, filters: Filters | None = None) -> list[Hit]:
        """Rank the chunks that pass ``filters`` and return the top ``k``."""
        filters = filters or Filters()
        query = list(dict.fromkeys(terms))  # unique, order kept
        scored: list[tuple[float, int]] = []
        for i, chunk in enumerate(self.chunks):
            if not filters.allows(chunk):
                continue
            tf, length, score = self.tf[i], len(self.docs[i]), 0.0
            for t in query:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    score += self.idf(t) * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [Hit(self.chunks[i], s, r + 1) for r, (s, i) in enumerate(scored[:k])]
