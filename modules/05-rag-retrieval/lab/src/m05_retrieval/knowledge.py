"""A knowledge-based index: product facts stored as (subject, predicate, object) triples.

Triples are extracted with rules from the tables in the help centre, so every
fact keeps a pointer to the document it came from. Lookups are exact: they
never "sort of" match, which is both their strength and their limit.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .parsing import Document

# Words in a question that point at a predicate (column or spec name).
PREDICATE_HINTS: dict[str, str] = {
    "weigh": "weight",
    "heavy": "weight",
    "lightest": "weight",
    "heaviest": "weight",
    "temperature": "operating temperature",
    "price": "price",
    "cost": "price",
    "cheapest": "price",
    "expensive": "price",
    "launch": "launched",
    "recent": "launched",
    "newest": "launched",
    "latest": "launched",
    "standby": "standby",
    "wi-fi": "wi-fi",
    "band": "wi-fi",
    "hub": "hub",
    "warranty": "warranty",
}
LOWEST = ("cheapest", "lightest")
HIGHEST = ("most recent", "newest", "latest", "heaviest", "most expensive")


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    doc_id: str


def product_name(doc: Document, entities: set[str]) -> str | None:
    """The product a product page is about: the known entity its title starts with."""
    return next((e for e in sorted(entities) if doc.title.startswith(e)), None)


def extract_triples(docs: list[Document]) -> list[Triple]:
    """Turn every table into triples.

    - A comparison table (first column = entity) gives (row entity, column header, cell).
    - A two-column ``Spec | Value`` table on a product page gives (product, spec, value).
    """
    triples: list[Triple] = []
    tables = [(d, b) for d in docs for b in d.blocks if b.kind == "table" and len(b.rows) > 1]
    entities = {row[0] for _, b in tables if b.rows[0][0].lower() == "device" for row in b.rows[1:]}
    for doc, table in tables:
        header, *rows = table.rows
        if tuple(h.lower() for h in header) == ("spec", "value"):
            subject = product_name(doc, entities)
            if subject:
                triples.extend(Triple(subject, r[0], r[1], doc.doc_id) for r in rows)
        elif header[0].lower() == "device":
            for r in rows:
                triples.extend(Triple(r[0], h, v, doc.doc_id) for h, v in zip(header[1:], r[1:]))
    return triples


class KnowledgeIndex:
    """subject -> triples, with a tiny rule-based question matcher."""

    def __init__(self, triples: list[Triple]) -> None:
        self.triples = triples
        self.by_subject: dict[str, list[Triple]] = defaultdict(list)
        for t in triples:
            self.by_subject[t.subject].append(t)

    def _predicate(self, question: str) -> str | None:
        q = question.lower()
        return next((p for hint, p in PREDICATE_HINTS.items() if hint in q), None)

    def lookup(self, question: str) -> list[Triple]:
        """Triples about the entity named in the question whose predicate matches a hint word."""
        q = question.lower()
        pred = self._predicate(question)
        subjects = [s for s in self.by_subject if s.lower() in q]
        if not pred:
            return []
        if subjects:
            return [t for s in subjects for t in self.by_subject[s] if pred in t.predicate.lower()]
        return self._superlative(q, pred)

    def _superlative(self, q: str, pred: str) -> list[Triple]:
        """'Which device is the cheapest / was launched most recently?' over one predicate."""
        direction = "min" if any(w in q for w in LOWEST) else "max" if any(w in q for w in HIGHEST) else ""
        if not direction:
            return []
        rows = [t for t in self.triples if pred in t.predicate.lower()]

        def key(t: Triple) -> tuple[float, str]:
            num = re.match(r"-?\d+(\.\d+)?$", t.object.strip("€ "))
            return (float(t.object.strip("€ ")) if num else 0.0, t.object)

        if not rows:
            return []
        best = min(rows, key=key) if direction == "min" else max(rows, key=key)
        return [best]
