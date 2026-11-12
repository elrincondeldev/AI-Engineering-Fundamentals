"""Chunkers, metadata filters and near-duplicate detection."""

from __future__ import annotations

import pytest

from m05_retrieval.chunking import (
    chunk_fixed,
    chunk_headings,
    chunk_recursive,
    chunk_semantic,
    chunk_table_aware,
    near_duplicates,
    prepare_corpus,
    row_to_text,
)
from m05_retrieval.parsing import Block, Document, load_kb
from m05_retrieval.text import word_count


def _doc(blocks: list[Block]) -> Document:
    return Document("kb-x", "Doc X", "x.md", "markdown", {"status": "current", "audience": "public"}, blocks)


def test_fixed_windows_overlap() -> None:
    doc = _doc([Block("text", " ".join(f"w{i}" for i in range(10)))])
    chunks = chunk_fixed(doc, size=4, overlap=2)
    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w2 w3 w4 w5", "w4 w5 w6 w7", "w6 w7 w8 w9"]
    with pytest.raises(ValueError):
        chunk_fixed(doc, size=4, overlap=4)


def test_recursive_respects_the_limit() -> None:
    long = ". ".join(f"Sentence number {i} is here" for i in range(30)) + "."
    doc = _doc([Block("text", long), Block("text", "Short tail.")])
    chunks = chunk_recursive(doc, max_words=25)
    assert all(word_count(c.text) <= 25 for c in chunks)
    assert "".join(c.text for c in chunks).replace("\n", " ").count("Sentence") == 30


def test_heading_chunks_follow_sections() -> None:
    doc = _doc(
        [Block("heading", "A", 1), Block("text", "one"), Block("heading", "B", 2), Block("text", "two")]
    )
    chunks = chunk_headings(doc)
    assert [(c.section, c.text) for c in chunks] == [("A", "A\none"), ("B", "B\ntwo")]
    assert [c.chunk_id for c in chunks] == ["kb-x#0", "kb-x#1"]


def test_table_aware_rows_keep_their_headers() -> None:
    rows = (("Device", "Weight (g)", "Price"), ("Watcher S1", "34", "29"))
    doc = _doc([Block("heading", "Compare", 1), Block("table", "flat", rows=rows)])
    chunks = chunk_table_aware(doc)
    assert chunks[-1].text == "Doc X | Watcher S1 | Weight (g): 34; Price: 29"
    assert row_to_text("T", ("A", "B"), ("x", "y")) == "T | x | B: y"


def test_two_column_tables_stay_inline() -> None:
    rows = (("Spec", "Value"), ("Weight", "34 g"))
    doc = _doc([Block("heading", "Specs", 2), Block("table", "Spec Value\nWeight 34 g", rows=rows)])
    assert [c.text for c in chunk_table_aware(doc)] == ["Specs\nSpec Value\nWeight 34 g"]


def test_semantic_chunker_is_deterministic_and_bounded() -> None:
    doc = next(d for d in load_kb() if d.doc_id == "kb-040")
    a, b = chunk_semantic(doc), chunk_semantic(doc)
    assert [c.text for c in a] == [c.text for c in b]
    assert all(word_count(c.text) <= 120 for c in a)


def test_filters_drop_superseded_internal_and_duplicates() -> None:
    docs = load_kb()
    kept, report = prepare_corpus(docs)
    assert report.dropped == {
        "kb-010": "status=superseded",
        "kb-041": "audience=internal",
        "kb-027": "near-duplicate of kb-026 (Jaccard 0.884)",
    }
    assert len(kept) == 23
    assert near_duplicates(docs) == [("kb-026", "kb-027", 0.884)]
