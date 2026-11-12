"""The note store: every piece of evidence, with its source, deduplicated and compacted."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date

from .providers import count_tokens


@dataclass
class Note:
    """One evidence passage, where it came from, and which sub-queries it answers."""

    url: str  # canonical URL of the page
    domain: str
    publisher: str
    title: str
    published: str
    text: str
    credibility: float
    recency: float
    relevance: dict[str, float]  # sub-query id -> share of its terms found in the passage
    mirrors: list[str] = field(default_factory=list)  # duplicate URLs merged into this note

    @property
    def owner(self) -> str:
        """The sub-query this note answers best (the first one on ties)."""
        return max(self.relevance, key=lambda sq: self.relevance[sq])

    @property
    def score(self) -> float:
        """Relevance x credibility x recency: how much the note deserves a place in the context."""
        return round(self.relevance[self.owner] * self.credibility * self.recency, 4)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


def content_key(text: str) -> str:
    """Hash of the normalised text: catches syndicated copies and tracking-URL variants."""
    norm = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def recency_weight(published: str, as_of: date, half_life_days: float) -> float:
    """Halve the weight of a source every ``half_life_days``; undated pages get 0.5."""
    try:
        age = (as_of - date.fromisoformat(published)).days
    except ValueError:
        return 0.5
    return round(0.5 ** (max(age, 0) / half_life_days), 4)


class NoteStore:
    """Holds notes keyed by content, so the same evidence is never stored twice."""

    def __init__(self, dedup: bool = True) -> None:
        self.dedup = dedup
        self._notes: dict[str, Note] = {}
        self._key_of_url: dict[str, str] = {}
        self.mirrors_merged = 0
        self.superseded: list[str] = []

    def add(self, note: Note) -> bool:
        """Add ``note``; return True if it is new evidence (not a duplicate).

        With dedup on, a note is a duplicate when its page (canonical URL) already gave a
        note, or when another page carries the same text (a mirror). The copy with the
        higher credibility is kept and the other URL is recorded as a mirror.
        """
        if not self.dedup:
            self._notes[f"{note.url}#{len(self._notes)}"] = note
            return True
        key = self._key_of_url.get(note.url) or content_key(note.text)
        old = self._notes.get(key)
        self._key_of_url[note.url] = key
        if old is None:
            self._notes[key] = note
            return True
        best, other = (note, old) if note.credibility > old.credibility else (old, note)
        if other.url != best.url:
            self.mirrors_merged += 1
            best.mirrors = sorted(set(best.mirrors + other.mirrors + [other.url]))
        for sq, rel in other.relevance.items():
            best.relevance[sq] = max(rel, best.relevance.get(sq, 0.0))
        self._notes[key] = best
        return False

    @property
    def notes(self) -> list[Note]:
        return list(self._notes.values())

    def for_subquery(self, sq_id: str) -> list[Note]:
        return [n for n in self._notes.values() if sq_id in n.relevance]

    def tokens(self) -> int:
        return sum(n.tokens for n in self._notes.values())

    def drop_superseded(self, subjects: list[str]) -> None:
        """Same publisher + same subject in the title: keep only the newest page."""
        for subject in subjects:
            groups: dict[str, list[tuple[str, Note]]] = {}
            for key, n in self._notes.items():
                if subject.lower() in n.title.lower():
                    groups.setdefault(n.publisher or n.domain, []).append((key, n))
            for items in groups.values():
                items.sort(key=lambda kn: kn[1].published, reverse=True)
                for key, n in items[1:]:
                    if n.published < items[0][1].published:
                        self.superseded.append(n.url)
                        del self._notes[key]

    def compact(self, per_subquery: int) -> int:
        """Keep the best ``per_subquery`` notes owned by each sub-query; return tokens freed."""
        before = self.tokens()
        by_owner: dict[str, list[tuple[str, Note]]] = {}
        for key, n in self._notes.items():
            by_owner.setdefault(n.owner, []).append((key, n))
        for items in by_owner.values():
            items.sort(key=lambda kn: (-kn[1].score, kn[1].url))
            for key, _ in items[per_subquery:]:
                del self._notes[key]
        return before - self.tokens()
