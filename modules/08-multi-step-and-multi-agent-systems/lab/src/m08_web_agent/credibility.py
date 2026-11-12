"""Transparent source-quality signals: a small trust list plus content-based spam checks.

Real search products combine many signals (link graphs, user feedback, editorial
lists). Here we keep three that you can read in one minute:

1. A reviewed allowlist of primary sources (standards bodies, research firms).
2. URL patterns for user-generated content (forums) and mirrors (syndicated copies).
3. Content signals of spam: keyword stuffing, shouting and sales phrases.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date

from .web import Page

# Primary sources the team reviewed and trusts for facts (an editable allowlist).
TRUSTED_DOMAINS = frozenset(
    {"homemesh.example", "eheo.example", "marketlens.example", "energyprices.example"}
)
MIRROR_PATHS = ("/syndicated/", "/copy/")
SALES_PHRASES = ("buy now", "limited offer", "limited deal", "90% off", "click here", "best deals")
TIER_SCORE = {"high": 1.0, "medium": 0.7, "low": 0.4, "spam": 0.05}

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class SpamSignals:
    """The evidence behind a spam decision, so a reviewer can audit it.

    A page is spam when at least two of three signals fire: a word trigram repeated
    5+ times (keyword stuffing), shouting (``!!!`` or a mostly upper-case title) and
    sales phrases. One signal alone is common on honest pages.
    """

    max_trigram_repeats: int
    shouting: bool
    sales_phrases: int

    @property
    def score(self) -> int:
        return int(self.max_trigram_repeats >= 5) + int(self.shouting) + int(self.sales_phrases >= 1)

    @property
    def is_spam(self) -> bool:
        return self.score >= 2


def spam_signals(page: Page) -> SpamSignals:
    """Measure keyword stuffing, shouting and sales language on one page."""
    return text_spam_signals(page.title, page.text)


def text_spam_signals(title: str, text: str) -> SpamSignals:
    """The same signals from raw strings (what a model can compute from an observation)."""
    words = [w.lower() for w in _WORD.findall(text)]
    trigrams = Counter(zip(words, words[1:], words[2:]))
    repeats = max(trigrams.values(), default=0)
    title_words = [w for w in _WORD.findall(title) if w.isalpha()]
    caps = sum(w.isupper() and len(w) > 1 for w in title_words)
    shouting = "!!!" in text or (caps >= 3 and caps / len(title_words) >= 0.5)
    lowered = f"{title}\n{text}".lower()
    sales = sum(lowered.count(p) for p in SALES_PHRASES)
    return SpamSignals(repeats, shouting, sales)


def source_tier(page: Page) -> str:
    """Return ``high``, ``medium``, ``low`` or ``spam`` for one page."""
    if spam_signals(page).is_spam:
        return "spam"
    if page.domain in TRUSTED_DOMAINS:
        return "high"
    if page.domain.startswith("forum.") or any(p in page.url for p in MIRROR_PATHS):
        return "low"
    return "medium"


def source_score(page: Page) -> float:
    """Tier as a number in (0, 1], used to re-rank search results."""
    return TIER_SCORE[source_tier(page)]


def recency_score(published: str, today: str = "2025-11-11") -> float:
    """1.0 for a page published today, falling linearly to 0.0 at three years old.

    ``today`` is fixed (the snapshot was crawled in November 2025) so runs are reproducible.
    """
    if not published:
        return 0.0
    age_days = (date.fromisoformat(today) - date.fromisoformat(published)).days
    return max(0.0, 1.0 - age_days / (3 * 365))
