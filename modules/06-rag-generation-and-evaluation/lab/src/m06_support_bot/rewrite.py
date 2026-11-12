"""Query rewriting: bridge the gap between the customer's words and the help centre's words.

Production systems usually ask an LLM to rewrite the query (fix typos, expand
synonyms, translate). Offline, we use a small curated glossary. Each word of the
question becomes a *concept*: the customer's word plus its synonyms and
translations. A concept is matched if any of its terms appears.
"""

from __future__ import annotations

from dataclasses import dataclass

from .text import content_tokens

# Written as plain words; ``_compile`` turns them into stemmed, accent-free tokens.
_GLOSSARY_WORDS: dict[str, tuple[str, ...]] = {
    # Customer vocabulary -> help-centre vocabulary
    "weigh": ("weight",),
    "cost": ("price",),
    "bright": ("lumens",),
    "disconnecting": ("dropping", "offline"),
    "subscribe": ("subscription",),
    # English -> Spanish (Latin America shipping is documented only in Spanish)
    "delivery": ("entrega", "plazo"),
    "shipped": ("envíos", "enviamos"),
    "customs": ("aduana", "aranceles"),
    "fees": ("tasas", "tarifa"),
    "Mexico": ("Latinoamérica",),
    # Spanish -> English
    "devolver": ("return",),
    "días": ("days",),
    "garantía": ("warranty",),
    "estándar": ("standard",),
}


def _compile(words: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for key, values in words.items():
        out[content_tokens(key)[0]] = tuple(t for v in values for t in content_tokens(v))
    return out


GLOSSARY: dict[str, tuple[str, ...]] = _compile(_GLOSSARY_WORDS)


@dataclass(frozen=True)
class Concept:
    """One word of the question with its alternatives. ``head`` is the customer's own word."""

    head: str
    terms: frozenset[str]


def concept_groups(question: str, glossary: dict[str, tuple[str, ...]] | None = None) -> list[Concept]:
    """Unique concepts of a question, in order."""
    glossary = GLOSSARY if glossary is None else glossary
    seen: dict[str, Concept] = {}
    for tok in content_tokens(question):
        if tok not in seen:
            seen[tok] = Concept(tok, frozenset({tok, *glossary.get(tok, ())}))
    return list(seen.values())


def rewrite_query(question: str, glossary: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """Flat list of search terms: the question's words followed by their expansions."""
    terms: list[str] = []
    for concept in concept_groups(question, glossary):
        terms.append(concept.head)
        terms.extend(sorted(concept.terms - {concept.head}))
    return terms
