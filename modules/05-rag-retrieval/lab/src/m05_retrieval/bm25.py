"""Keyword search: an inverted index scored with BM25."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .text import tokenize


@dataclass
class InvertedIndex:
    """term -> {chunk position: term frequency}, plus the lengths BM25 needs."""

    postings: dict[str, dict[int, int]]
    lengths: list[int]

    @classmethod
    def build(cls, texts: list[str], *, fold: bool = False) -> InvertedIndex:
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        lengths = []
        for pos, text in enumerate(texts):
            counts = Counter(tokenize(text, fold=fold))
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                postings[term][pos] = tf
        return cls(dict(postings), lengths)

    @property
    def n_docs(self) -> int:
        return len(self.lengths)

    @property
    def n_postings(self) -> int:
        return sum(len(p) for p in self.postings.values())

    def boolean_and(self, terms: list[str]) -> set[int]:
        """Chunks that contain every term (the oldest kind of keyword search)."""
        sets = [set(self.postings.get(t, {})) for t in terms]
        return set.intersection(*sets) if sets else set()


def idf(n_docs: int, df: int) -> float:
    """BM25 inverse document frequency (the '+1' keeps it positive for very common terms)."""
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


class BM25:
    """Okapi BM25 over an :class:`InvertedIndex`. ``k1`` caps term frequency, ``b`` normalises length."""

    def __init__(self, texts: list[str], *, k1: float = 1.5, b: float = 0.75, fold: bool = False) -> None:
        self.k1, self.b, self.fold = k1, b, fold
        self.index = InvertedIndex.build(texts, fold=fold)
        self.avg_len = sum(self.index.lengths) / max(1, self.index.n_docs)

    def term_score(self, tf: int, length: int, term_idf: float) -> float:
        """Contribution of one query term to one chunk."""
        norm = 1 - self.b + self.b * length / self.avg_len
        return term_idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)

    def scores(self, query: str) -> list[float]:
        """Score every chunk; only chunks in the query terms' posting lists are touched."""
        out = [0.0] * self.index.n_docs
        for term in tokenize(query, fold=self.fold):
            plist = self.index.postings.get(term)
            if not plist:
                continue
            term_idf = idf(self.index.n_docs, len(plist))
            for pos, tf in plist.items():
                out[pos] += self.term_score(tf, self.index.lengths[pos], term_idf)
        return out
