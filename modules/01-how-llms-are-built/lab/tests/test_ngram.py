from __future__ import annotations

from m01_foundations.ngram import DOC_BREAK, NGramModel, corpus_tokens, coverage, split_docs, words


def test_words_lowercases_and_splits_punctuation() -> None:
    assert words("Turn it OFF, now.") == ["turn", "it", "off", ",", "now", "."]


def test_next_word_probabilities_and_backoff() -> None:
    model = NGramModel(order=3).fit(corpus_tokens(["the plug is on", "the plug is off", "a bulb is on"]))
    used, probs = model.next_word_probs(words("the plug is"))
    assert used == 2
    assert probs == [("off", 0.5), ("on", 0.5)]
    used, probs = model.next_word_probs(words("my lamp is"))  # 'lamp is' unseen: back off to 'is'
    assert used == 1
    assert probs[0] == ("on", 2 / 3)
    assert model.next_word_probs(["zebra"]) == (0, [])


def test_generate_stops_at_document_break() -> None:
    model = NGramModel(order=2).fit(corpus_tokens(["switch the light off"]))
    assert model.generate("switch") == ["the", "light", "off"]
    assert DOC_BREAK not in model.generate("light")


def test_coverage_falls_as_n_grows() -> None:
    train = corpus_tokens(["the plug is on in the kitchen", "the bulb is off in the hallway"])
    held = corpus_tokens(["the plug is off in the hallway"])
    values = [coverage(train, held, n) for n in range(1, 5)]
    assert values[0] == 1.0
    assert values == sorted(values, reverse=True)


def test_split_docs_is_deterministic() -> None:
    train, held = split_docs([str(i) for i in range(10)])
    assert held == ["0", "5"]
    assert len(train) == 8
