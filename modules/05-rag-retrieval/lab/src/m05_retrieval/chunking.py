"""Chunking strategies, metadata filters and near-duplicate detection.

Every chunker takes a parsed :class:`~m05_retrieval.parsing.Document` and
returns a list of :class:`Chunk`. They differ only in *where they cut*.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from .parsing import Block, Document
from .providers import hashing_embedding
from .text import split_sentences, tokenize, word_count


@dataclass(frozen=True)
class Chunk:
    """A piece of a document that is indexed and retrieved on its own."""

    chunk_id: str
    doc_id: str
    text: str
    section: str = ""
    meta: dict[str, str] = field(default_factory=dict, compare=False, hash=False)


def _make(doc: Document, pieces: Iterable[tuple[str, str]]) -> list[Chunk]:
    out = []
    for section, text in pieces:
        if text.strip():
            out.append(Chunk(f"{doc.doc_id}#{len(out)}", doc.doc_id, text.strip(), section, doc.meta))
    return out


# --------------------------------------------------------------------------- #
# 1. Fixed-size windows with overlap                                          #
# --------------------------------------------------------------------------- #


def chunk_fixed(doc: Document, size: int = 80, overlap: int = 20) -> list[Chunk]:
    """Cut the whole text into windows of ``size`` words; consecutive windows share ``overlap`` words."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    words = doc.text.split()
    step = size - overlap
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(("", " ".join(words[start : start + size])))
        if start + size >= len(words):
            break
    return _make(doc, pieces)


# --------------------------------------------------------------------------- #
# 2. Recursive splitting                                                      #
# --------------------------------------------------------------------------- #


def _split_recursive(text: str, max_words: int) -> list[str]:
    """Split by paragraph, then sentence, then words, until every piece fits."""
    if word_count(text) <= max_words:
        return [text]
    for splitter in (lambda t: t.split("\n"), split_sentences):
        parts = [p for p in splitter(text) if p.strip()]
        if len(parts) > 1:
            return [piece for p in parts for piece in _split_recursive(p, max_words)]
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def chunk_recursive(doc: Document, max_words: int = 100) -> list[Chunk]:
    """Split recursively, then greedily merge neighbours back together up to ``max_words``."""
    pieces = _split_recursive(doc.text, max_words)
    merged: list[str] = []
    for piece in pieces:
        if merged and word_count(merged[-1]) + word_count(piece) <= max_words:
            merged[-1] = f"{merged[-1]}\n{piece}"
        else:
            merged.append(piece)
    return _make(doc, (("", m) for m in merged))


# --------------------------------------------------------------------------- #
# 3. Structure-aware: one chunk per heading section                           #
# --------------------------------------------------------------------------- #


def _sections(doc: Document, max_level: int = 3) -> list[tuple[str, list[Block]]]:
    """Group blocks under the most recent heading of level <= ``max_level``."""
    sections: list[tuple[str, list[Block]]] = []
    for block in doc.blocks:
        if block.kind == "heading" and block.level <= max_level:
            sections.append((block.text, [block]))
        elif sections:
            sections[-1][1].append(block)
        else:
            sections.append(("", [block]))
    return sections


def chunk_headings(doc: Document) -> list[Chunk]:
    """One chunk per heading section; tables are flattened into the section text."""
    return _make(doc, ((title, "\n".join(b.text for b in blocks)) for title, blocks in _sections(doc)))


# --------------------------------------------------------------------------- #
# 4. Table-aware: heading sections + one self-describing chunk per table row  #
# --------------------------------------------------------------------------- #


def row_to_text(title: str, header: tuple[str, ...], row: tuple[str, ...]) -> str:
    """Serialise one table row so that every value keeps its column name."""
    pairs = "; ".join(f"{h}: {v}" for h, v in zip(header[1:], row[1:]))
    return f"{title} | {row[0]} | {pairs}"


def chunk_table_aware(doc: Document, min_columns: int = 3) -> list[Chunk]:
    """Like :func:`chunk_headings`, but rows of wide tables become chunks that keep their headers.

    Two-column key/value tables read fine when flattened, so they stay in the section text.
    """
    pieces: list[tuple[str, str]] = []
    for title, blocks in _sections(doc):
        wide = [b for b in blocks if b.kind == "table" and len(b.rows[0]) >= min_columns]
        pieces.append((title, "\n".join(b.text for b in blocks if b not in wide)))
        for table in wide:
            header, *rows = table.rows
            pieces.extend((title, row_to_text(doc.title, header, row)) for row in rows)
    return _make(doc, pieces)


# --------------------------------------------------------------------------- #
# 5. Semantic: cut where consecutive sentences stop being similar             #
# --------------------------------------------------------------------------- #


def chunk_semantic(
    doc: Document, threshold: float = 0.05, min_words: int = 25, max_words: int = 120
) -> list[Chunk]:
    """Embed every sentence; start a new chunk where similarity to the previous sentence drops.

    A chunk always grows to ``min_words`` first and never exceeds ``max_words``.
    """
    sentences = [s for b in doc.blocks for s in (split_sentences(b.text) or [b.text])]
    if not sentences:
        return []
    vecs = np.array([hashing_embedding(s) for s in sentences])
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        size = word_count(" ".join(groups[-1]))
        topic_shift = float(vecs[i] @ vecs[i - 1]) < threshold
        too_big = size + word_count(sentences[i]) > max_words
        if too_big or (topic_shift and size >= min_words):
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])
    return _make(doc, (("", "\n".join(g)) for g in groups))


CHUNKERS: dict[str, Callable[[Document], list[Chunk]]] = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "heading": chunk_headings,
    "table": chunk_table_aware,
    "semantic": chunk_semantic,
}


def with_title(chunks: list[Chunk], docs: dict[str, Document]) -> list[Chunk]:
    """Prefix each chunk with its document title (a 'contextual header')."""
    return [
        Chunk(c.chunk_id, c.doc_id, f"{docs[c.doc_id].title}\n{c.text}", c.section, c.meta) for c in chunks
    ]


# --------------------------------------------------------------------------- #
# Metadata filtering and de-duplication                                       #
# --------------------------------------------------------------------------- #


def is_servable(doc: Document, *, audience: str = "public") -> bool:
    """A customer-facing bot may only see current documents for its audience."""
    return doc.meta.get("status") == "current" and doc.meta.get("audience") == audience


def shingles(text: str, k: int = 5) -> set[str]:
    words = tokenize(text)
    return {" ".join(words[i : i + k]) for i in range(max(1, len(words) - k + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def near_duplicates(docs: list[Document], threshold: float = 0.8) -> list[tuple[str, str, float]]:
    """Return (kept_id, duplicate_id, similarity) for document pairs above ``threshold``.

    The first document in manifest order is kept; later copies are reported as duplicates.
    """
    sh = [shingles(d.text) for d in docs]
    pairs = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sim = jaccard(sh[i], sh[j])
            if sim >= threshold:
                pairs.append((docs[i].doc_id, docs[j].doc_id, round(sim, 3)))
    return pairs


@dataclass(frozen=True)
class CorpusReport:
    """What the ingestion filters removed, for the ``chunk`` command."""

    kept: list[str]
    dropped: dict[str, str]  # doc_id -> reason


def prepare_corpus(
    docs: list[Document], *, filter_meta: bool = True, dedupe: bool = True
) -> tuple[list[Document], CorpusReport]:
    """Apply metadata filters and near-duplicate removal before chunking."""
    dropped: dict[str, str] = {}
    if filter_meta:
        for d in docs:
            if d.meta.get("audience") != "public":
                dropped[d.doc_id] = f"audience={d.meta.get('audience')}"
            elif d.meta.get("status") != "current":
                dropped[d.doc_id] = f"status={d.meta.get('status')}"
    if dedupe:
        candidates = [d for d in docs if d.doc_id not in dropped]
        for kept, dup, sim in near_duplicates(candidates):
            dropped.setdefault(dup, f"near-duplicate of {kept} (Jaccard {sim})")
    kept_docs = [d for d in docs if d.doc_id not in dropped]
    return kept_docs, CorpusReport([d.doc_id for d in kept_docs], dropped)


def chunk_corpus(docs: list[Document], chunker: str = "heading", *, title: bool = False) -> list[Chunk]:
    """Chunk every document with the named strategy."""
    fn = CHUNKERS[chunker]
    chunks = [c for d in docs for c in fn(d)]
    return with_title(chunks, {d.doc_id: d for d in docs}) if title else chunks
