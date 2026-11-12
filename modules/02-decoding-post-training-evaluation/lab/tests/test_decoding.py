"""Unit tests for the n-gram model and every decoding building block."""

from __future__ import annotations

import math

import numpy as np
import pytest

from m02_post_training.decoding import (
    DecodeConfig,
    apply_frequency_penalty,
    apply_repetition_penalty,
    beam_search,
    distinct_n,
    generate,
    repetition_rate,
    softmax,
    top_k_filter,
    top_p_filter,
)
from m02_post_training.ngram import EOS, NgramLM
from m02_post_training.text import detokenize, tokenize

TOY = [
    "the hub is on . the hub is off .".split(),
    "the bulb is on . the plug is on .".split(),
    "reset the hub . reset the bulb .".split(),
]


@pytest.fixture(scope="module")
def toy_lm() -> NgramLM:
    return NgramLM(min_count=1).fit(TOY)


def test_tokenize_and_detokenize_round_trip() -> None:
    toks = tokenize("Wi-Fi 2.4 GHz only, (not 5 GHz).")
    assert toks == ["wi-fi", "2.4", "ghz", "only", ",", "(", "not", "5", "ghz", ")", "."]
    assert detokenize(toks) == "wi-fi 2.4 ghz only, (not 5 ghz)."


def test_ngram_distribution_sums_to_one(toy_lm: NgramLM) -> None:
    for ctx in ([], ["the"], ["the", "hub"], ["never", "seen"]):
        assert math.isclose(float(np.exp(toy_lm.next_logprobs(ctx)).sum()), 1.0, rel_tol=1e-9)


def test_ngram_prefers_seen_continuations(toy_lm: NgramLM) -> None:
    lp = toy_lm.next_logprobs(["the", "hub"])
    assert toy_lm.vocab[int(np.argmax(lp))] == "is"


def test_perplexity_is_lower_on_training_text(toy_lm: NgramLM) -> None:
    assert toy_lm.perplexity(TOY) < toy_lm.perplexity([["off", "bulb", "reset", "plug"]])


def test_softmax_and_temperature() -> None:
    logits = np.array([2.0, 1.0, 0.0])
    p1 = softmax(logits, 1.0)
    assert math.isclose(p1.sum(), 1.0)
    assert softmax(logits, 0.5)[0] > p1[0] > softmax(logits, 2.0)[0]  # low T sharpens, high T flattens
    with pytest.raises(ValueError):
        softmax(logits, 0.0)


def test_top_k_keeps_exactly_k() -> None:
    out = top_k_filter(np.array([0.1, 3.0, 2.0, -1.0]), 2)
    assert list(np.isfinite(out)) == [False, True, True, False]


def test_top_p_keeps_smallest_set_reaching_p() -> None:
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    logits = np.log(probs)
    assert np.isfinite(top_p_filter(logits, 0.5)).sum() == 1
    assert np.isfinite(top_p_filter(logits, 0.8)).sum() == 2
    assert np.isfinite(top_p_filter(logits, 0.81)).sum() == 3
    assert np.isfinite(top_p_filter(logits, 1.0)).sum() == 4


def test_repetition_penalty_always_lowers_seen_tokens() -> None:
    logits = np.array([2.0, -2.0, 1.0])
    out = apply_repetition_penalty(logits, [0, 1], 2.0)
    assert out[0] == 1.0 and out[1] == -4.0 and out[2] == 1.0


def test_frequency_and_presence_penalty() -> None:
    out = apply_frequency_penalty(np.zeros(3), [0, 0, 1], frequency=0.5, presence=1.0)
    assert list(out) == [-2.0, -1.5, 0.0]


def test_greedy_is_deterministic_and_seeds_reproduce(toy_lm: NgramLM) -> None:
    g1 = generate(toy_lm, "the", DecodeConfig(strategy="greedy", max_new_tokens=8))
    g2 = generate(toy_lm, "the", DecodeConfig(strategy="greedy", max_new_tokens=8, seed=123))
    assert g1.tokens == g2.tokens
    s1 = generate(toy_lm, "the", DecodeConfig(seed=7, max_new_tokens=12))
    s2 = generate(toy_lm, "the", DecodeConfig(seed=7, max_new_tokens=12))
    assert s1.tokens == s2.tokens


def test_max_tokens_and_stop_sequences(toy_lm: NgramLM) -> None:
    g = generate(toy_lm, "the", DecodeConfig(seed=0, max_new_tokens=3, temperature=0.01))
    assert len(g.tokens) <= 3
    stopped = generate(toy_lm, "the", DecodeConfig(strategy="greedy", max_new_tokens=20, stop=(".",)))
    assert stopped.finish_reason == "stop"
    assert "." not in stopped.text


def test_beam_with_one_beam_matches_greedy(toy_lm: NgramLM) -> None:
    greedy = generate(toy_lm, "reset", DecodeConfig(strategy="greedy", max_new_tokens=10))
    beam = beam_search(toy_lm, "reset", DecodeConfig(strategy="beam", num_beams=1, max_new_tokens=10))
    assert beam.tokens == greedy.tokens


def test_generation_never_emits_special_tokens(toy_lm: NgramLM) -> None:
    for seed in range(5):
        g = generate(toy_lm, "the", DecodeConfig(seed=seed, temperature=3.0, max_new_tokens=30))
        assert not {"<bos>", "<unk>", EOS} & set(g.tokens)


def test_distinct_n_and_repetition_rate() -> None:
    assert distinct_n([["a", "b", "a", "b"]], 2) == pytest.approx(2 / 3)
    assert distinct_n([["a", "b"], ["a", "b"]], 2) == 0.5
    assert repetition_rate(["a", "b", "c", "a", "b", "c"], 3) == pytest.approx(1 / 4)
    assert repetition_rate(["a", "b"], 3) == 0.0
