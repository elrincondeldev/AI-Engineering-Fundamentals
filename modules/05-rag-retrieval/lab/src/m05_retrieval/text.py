"""Tokenisation helpers shared by the keyword index, the chunkers and the reranker."""

from __future__ import annotations

import re
import unicodedata

WORD_RE = re.compile(r"\w+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9¿¡ÁÉÍÓÚÑ])")

# A tiny English + Spanish stop-word list. BM25 does not need it (IDF already
# down-weights common words), but the lexical reranker does.
STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its my of on or so
    that the their there this to was what when where which who why will with you your
    de del el en es la las los para por que un una y""".split()
)


def fold_accents(text: str) -> str:
    """Remove diacritics: 'México' -> 'Mexico'. Off by default; see the lab's Extend it section."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str, *, fold: bool = False) -> list[str]:
    """Lower-case word tokens. The same rule is used for documents and queries."""
    if fold:
        text = fold_accents(text)
    return [w.lower() for w in WORD_RE.findall(text)]


def content_terms(text: str) -> list[str]:
    """Tokens without stop words, in order, without repeats."""
    seen: list[str] = []
    for tok in tokenize(text):
        if tok not in STOPWORDS and tok not in seen:
            seen.append(tok)
    return seen


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences with a simple punctuation rule (good enough for help articles)."""
    parts = [p.strip() for p in SENTENCE_RE.split(text.strip())]
    return [p for p in parts if p]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))
