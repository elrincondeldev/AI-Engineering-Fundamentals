from __future__ import annotations

import pytest

from m01_foundations.bpe import (
    END_OF_TEXT,
    BPETokenizer,
    bytes_per_token,
    merge_pair,
    pair_counts,
    pretokenize,
    tokens_per_word,
    train_bpe,
)

CORPUS = ["low lower lowest", "new newer newest", "the lowest price is the newest price"] * 3


def test_pretokenize_keeps_leading_space_with_word() -> None:
    assert pretokenize("Hi there, it's 2025!") == ["Hi", " there", ",", " it", "'s", " 202", "5", "!"]


def test_pair_counts_weights_by_frequency() -> None:
    counts = pair_counts({(1, 2, 3): 2, (2, 3): 5})
    assert counts[(2, 3)] == 7
    assert counts[(1, 2)] == 2


def test_merge_pair_left_to_right_without_overlap() -> None:
    assert merge_pair((1, 1, 1, 2), (1, 1), 9) == (9, 1, 2)
    assert merge_pair((5, 6, 5, 6), (5, 6), 7) == (7, 7)
    assert merge_pair((5,), (5, 6), 7) == (5,)


def test_training_learns_the_most_frequent_pair_first() -> None:
    tok, history = train_bpe(CORPUS, vocab_size=256 + 10 + 1)
    assert len(tok.merges) == 10
    # 'we' occurs in lower, lowest (x2), newer and newest (x2): 6 times per copy, more than any other pair
    assert history[0].piece == b"we"
    assert history[0].count == 18
    assert tok.vocab_size == 267
    assert tok.special == {END_OF_TEXT: 266}


@pytest.mark.parametrize(
    "text", ["", "plain ascii", "¿Cuánto tarda el envío?", "Ünïcödé 🙂 ½ 中文", "a\n\n  b\t"]
)
def test_round_trip_any_text(text: str) -> None:
    tok, _ = train_bpe(CORPUS, vocab_size=300)
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_are_atomic() -> None:
    tok, _ = train_bpe(CORPUS, vocab_size=280)
    ids = tok.encode(f"low{END_OF_TEXT}new")
    assert ids.count(tok.special[END_OF_TEXT]) == 1
    assert tok.decode(ids) == f"low{END_OF_TEXT}new"
    assert tok.special[END_OF_TEXT] not in tok.encode(f"low{END_OF_TEXT}", allow_special=False)


def test_more_merges_means_fewer_tokens() -> None:
    tok, _ = train_bpe(CORPUS, vocab_size=300)
    small = tok.truncated(5)
    text = "the lowest newest price"
    assert len(tok.encode(text)) < len(small.encode(text)) < len(BPETokenizer().encode(text))
    assert small.decode(small.encode(text)) == text


def test_byte_tokens_are_shown_as_hex() -> None:
    tok = BPETokenizer()
    assert tok.token_strings(tok.encode("ñ")) == ["<C3>", "<B1>"]


def test_metrics() -> None:
    raw = BPETokenizer()  # bytes only
    assert tokens_per_word(raw, ["ab cd"]) == 2.5
    assert bytes_per_token(raw, ["ab cd"]) == 1.0


def test_vocab_size_too_small() -> None:
    with pytest.raises(ValueError):
        train_bpe(CORPUS, vocab_size=200)
