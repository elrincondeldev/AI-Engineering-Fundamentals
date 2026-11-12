"""BM25 search over the snapshot, with optional dedup, credibility and recency re-ranking.

``search_web`` in the agent is a thin wrapper around :class:`SearchEngine`.
Each guard can be switched off to measure what it is worth (see ``lab ablate``).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from .credibility import TIER_SCORE, recency_score, source_tier
from .web import Page, Snapshot

_WORD = re.compile(r"\w+", re.UNICODE)
STOPWORDS = frozenset(
    "a an and are as at be by can did do does for from has have how in is it its of on or "
    "the than that this to was were what when which who why will with".split()
)
RECENCY_CUES = frozenset({"latest", "current", "currently", "now", "newest", "recent", "updated", "update"})


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without stopwords."""
    return [w for w in (t.lower() for t in _WORD.findall(text)) if w not in STOPWORDS]


def url_words(url: str) -> str:
    """Words in a URL ('https://eheo.example/guidelines/standby-power' -> 'eheo guidelines standby power')."""
    return " ".join(
        w for w in re.findall(r"[a-z0-9]+", url.lower()) if w not in {"https", "http", "www", "example"}
    )


class BM25:
    """Okapi BM25 over pre-tokenised documents (k1=1.5, b=0.75)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.lengths = [len(d) for d in docs]
        self.avg = sum(self.lengths) / max(len(docs), 1)
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: list[str], i: int) -> float:
        tf, norm = self.tf[i], self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
        return sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf)


def shingles(text: str, k: int = 5) -> set[str]:
    words = [w.lower() for w in _WORD.findall(text)]
    return {" ".join(words[i : i + k]) for i in range(max(len(words) - k + 1, 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def duplicate_groups(pages: list[Page], threshold: float = 0.8) -> dict[str, str]:
    """Map every URL to the URL of its group representative.

    Pages are grouped when they share a canonical URL (tracking variants), have the
    same article text (syndicated copies) or a 5-gram Jaccard similarity of at least
    ``threshold`` (lightly edited copies). The representative is the most credible
    page, then the shortest URL.
    """
    parent = {p.url: p.url for p in pages}

    def find(u: str) -> str:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    by_key: dict[str, str] = {}
    for p in pages:
        for key in (p.canonical, hashlib.sha1(" ".join(p.text.split()).encode()).hexdigest()):
            if key in by_key:
                union(p.url, by_key[key])
            else:
                by_key[key] = p.url
    sh = {p.url: shingles(p.text) for p in pages}
    for i, a in enumerate(pages):
        for b in pages[i + 1 :]:
            if find(a.url) != find(b.url) and jaccard(sh[a.url], sh[b.url]) >= threshold:
                union(a.url, b.url)
    groups: dict[str, list[Page]] = {}
    for p in pages:
        groups.setdefault(find(p.url), []).append(p)
    rep: dict[str, str] = {}
    for members in groups.values():
        best = min(members, key=lambda p: (-TIER_SCORE[source_tier(p)], len(p.url), p.url))
        rep.update({m.url: best.url for m in members})
    return rep


@dataclass(frozen=True)
class Guards:
    """Harness safeguards. Switch one off to measure what it is worth."""

    dedup: bool = True
    credibility: bool = True
    recency: bool = True
    sanitize: bool = True

    @classmethod
    def none(cls) -> Guards:
        return cls(dedup=False, credibility=False, recency=False, sanitize=False)

    def label(self) -> str:
        off = [name for name in ("dedup", "credibility", "recency", "sanitize") if not getattr(self, name)]
        return "all guards on" if not off else "no " + ", no ".join(off)


@dataclass(frozen=True)
class Hit:
    """One search result as the agent sees it."""

    rank: int
    url: str
    title: str
    published: str
    tier: str  # empty when the credibility guard is off
    score: float
    snippet: str


class SearchEngine:
    """BM25 over title + article text of every readable page."""

    def __init__(self, snapshot: Snapshot, guards: Guards = Guards()) -> None:
        self.guards = guards
        pages = snapshot.readable
        self.rep = duplicate_groups(pages) if guards.dedup else {p.url: p.url for p in pages}
        self.pages = [p for p in pages if self.rep[p.url] == p.url]
        self.bm25 = BM25([tokenize(f"{p.title} {p.text} {url_words(p.url)}") for p in self.pages])
        self.tiers = [source_tier(p) for p in self.pages]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = tokenize(query)
        wants_recent = bool(RECENCY_CUES & set(q))
        scored: list[tuple[float, int]] = []
        for i, page in enumerate(self.pages):
            s = self.bm25.score(q, i)
            if s <= 0:
                continue
            if self.guards.credibility:
                s *= TIER_SCORE[self.tiers[i]]
            if self.guards.recency:
                weight = 0.5 if wants_recent else 0.15
                s *= (1 - weight) + weight * recency_score(page.published)
            scored.append((s, i))
        scored.sort(key=lambda t: (-t[0], self.pages[t[1]].url))
        hits = []
        for rank, (s, i) in enumerate(scored[:k], 1):
            p = self.pages[i]
            tier = self.tiers[i] if self.guards.credibility else ""
            hits.append(Hit(rank, p.url, p.title, p.published, tier, round(s, 3), _snippet(p.text, q)))
        return hits


def _snippet(text: str, query: list[str], width: int = 160) -> str:
    """The sentence with the most query words, cut to ``width`` characters."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]
    if not sentences:
        return ""
    qs = set(query)
    best = max(sentences, key=lambda s: (len(qs & set(tokenize(s))), -sentences.index(s)))
    return best if len(best) <= width else best[: width - 3] + "..."
