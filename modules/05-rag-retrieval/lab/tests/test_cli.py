"""The CLI commands run and print what the lab README shows."""

from __future__ import annotations

import pytest

from m05_retrieval.cli import main


def test_parse(capsys: pytest.CaptureFixture[str]) -> None:
    main(["parse"])
    out = capsys.readouterr().out
    assert "26 documents, 13 tables (80 data rows), 1 hidden HTML element(s) dropped" in out


def test_chunk_writes_outputs(capsys: pytest.CaptureFixture[str]) -> None:
    main(["chunk", "--chunker", "table", "--show", "kb-007"])
    out = capsys.readouterr().out
    assert "23 documents -> 128 chunks" in out
    assert "Compare Hearthwise devices | Watcher S1 | Price (EUR): 29;" in out


def test_search_and_kg(capsys: pytest.CaptureFixture[str]) -> None:
    main(["search", "Who pays customs fees for orders shipped to Mexico?", "--fold"])
    assert "kb-031#" in capsys.readouterr().out
    main(["kg", "How much does the Watcher S1 weigh?"])
    assert "(Watcher S1, Weight (g), 34)  source: kb-007" in capsys.readouterr().out


def test_single_config_eval(capsys: pytest.CaptureFixture[str]) -> None:
    main(["eval", "--chunker", "heading"])
    out = capsys.readouterr().out
    assert "misses at 5: q004, q017, q018, q022" in out


def test_small_ann_without_faiss(capsys: pytest.CaptureFixture[str]) -> None:
    main(["ann", "--n", "3000", "--no-faiss"])
    out = capsys.readouterr().out
    assert "numpy exact" in out and "faiss" not in out.split("\n", 3)[3]
