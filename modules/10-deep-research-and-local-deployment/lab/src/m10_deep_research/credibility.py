"""Source credibility: a curated source policy plus content signals.

The agent never reads the ground-truth labels. It combines two things a real
team would have:

1. A **source policy** that your organisation maintains (trusted research
   bodies, known publishers, sites to treat as user-generated content).
2. **Content signals** that catch unknown sites: keyword stuffing, shouting,
   sales language and forum-style posts.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .corpus import Page

# Prior trust for domains the team has reviewed. Anything else starts at DEFAULT_PRIOR.
SOURCE_POLICY: dict[str, float] = {
    "marketlens.example": 0.9,  # independent market research
    "eheo.example": 0.9,  # energy research observatory
    "energyprices.example": 0.9,  # price statistics
    "homemesh.example": 0.9,  # standards body
    "smarthomeweekly.example": 0.7,  # established tech press
    "logistics-news.example": 0.7,
    "hogarconectado.example": 0.7,
    "www.hearthwise.example": 0.6,  # vendors: fine for their own prices and specs
    "corvid-home.example": 0.6,
    "brightline.example": 0.6,
    "tessera.example": 0.6,
}
DEFAULT_PRIOR = 0.5

_SALES = re.compile(
    r"\b(buy now|limited (?:deal|offer)|click here|best .{0,30} deals?|discount|90% off)\b", re.I
)
_FORUM_POST = re.compile(r"^user [\w.-]+:", re.I)
_WORDS = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class Credibility:
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


def stuffing_ratio(text: str) -> float:
    """Share of word trigrams taken by the most repeated trigram, if it repeats 3+ times."""
    words = [w.lower() for w in _WORDS.findall(text)]
    trigrams = list(zip(words, words[1:], words[2:]))
    top = Counter(trigrams).most_common(1)[0][1] if trigrams else 0
    return top / len(trigrams) if top >= 3 else 0.0


def shouting_ratio(text: str) -> float:
    """Share of words (3+ letters) written in capitals."""
    words = [w for w in _WORDS.findall(text) if len(w) >= 3]
    return sum(w.isupper() for w in words) / len(words) if words else 0.0


def score_page(page: Page) -> Credibility:
    """Return a credibility score in [0, 1] with the reasons that moved it."""
    score = SOURCE_POLICY.get(page.domain, DEFAULT_PRIOR)
    reasons = [f"policy prior {score:.1f}" if page.domain in SOURCE_POLICY else f"unknown domain {score:.1f}"]
    text = f"{page.title} {page.text}"
    if stuffing_ratio(text) > 0.08:
        score -= 0.3
        reasons.append("keyword stuffing")
    if shouting_ratio(text) > 0.15 or "!!!" in text:
        score -= 0.2
        reasons.append("shouting")
    if _SALES.search(text):
        score -= 0.2
        reasons.append("sales language")
    if sum(bool(_FORUM_POST.match(p)) for p in page.paragraphs) >= 2:
        score = min(score, 0.3)
        reasons.append("user-generated forum posts")
    return Credibility(round(max(score, 0.0), 2), tuple(reasons))
