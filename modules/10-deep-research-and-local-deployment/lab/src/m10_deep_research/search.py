"""A small BM25 search engine over the offline snapshot (standard library only)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import Page

_WORD_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[.,][0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was were what "
    "which with does do should there than more first next".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers ("0.4", "260,000" stay whole); stopwords removed."""
    return [t for t in _WORD_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    url: str
    score: float


class BM25Index:
    """Okapi BM25 over whole pages (title counted twice, then the article text)."""

    def __init__(self, pages: list[Page], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.urls = [p.url for p in pages]
        self.tfs: list[Counter[str]] = []
        self.lengths: list[int] = []
        df: Counter[str] = Counter()
        for p in pages:
            tokens = tokenize(f"{p.title} {p.title} {p.text}")
            tf = Counter(tokens)
            self.tfs.append(tf)
            self.lengths.append(len(tokens))
            df.update(tf.keys())
        n = len(pages)
        self.avg_len = sum(self.lengths) / max(n, 1)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Return the top ``k`` pages for ``query``, best first (ties broken by URL)."""
        terms = tokenize(query)
        scored: list[Hit] = []
        for url, tf, length in zip(self.urls, self.tfs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            s = sum(self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            if s > 0:
                scored.append(Hit(url, round(s, 6)))
        scored.sort(key=lambda h: (-h.score, h.url))
        return scored[:k]
