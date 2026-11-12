"""Unit tests for metrics, contamination, Elo / Bradley–Terry and the judge diagnostics."""

from __future__ import annotations

import pytest

from m02_post_training.evaluation import (
    EvalItem,
    Vote,
    biased_judge_responder,
    bleu,
    bradley_terry_ratings,
    elo_ratings,
    exact_match,
    find_contamination,
    judge_swap_test,
    parse_verdict,
    rouge_l,
    simulate_votes,
    strip_padding,
    token_f1,
)
from m02_post_training.providers import FakeProvider


def test_exact_match_normalises() -> None:
    assert exact_match("The 30 days.", "30 days") == 1.0
    assert exact_match("30 days", "14 days") == 0.0


def test_token_f1() -> None:
    assert token_f1("30 days from delivery", "30 days from delivery") == 1.0
    assert token_f1("30 days", "30 days from delivery") == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert token_f1("nothing shared", "30 days") == 0.0


def test_bleu_and_rouge_l() -> None:
    assert bleu("the hub is on", "the hub is on") == pytest.approx(1.0)
    assert bleu("completely different words", "the hub is on") == 0.0
    assert rouge_l("the hub is on", "the hub is on") == 1.0
    assert rouge_l("hub the on", "the hub is on") == pytest.approx(2 * (2 / 3) * (2 / 4) / (2 / 3 + 2 / 4))


def test_find_contamination_on_toy_pages() -> None:
    items = [EvalItem("x", "q1", tuple("how long is the warranty on the hub ?".split()))]
    pages = {"copy": "Quiz: how long is the warranty on the hub? Two years.", "clean": "The hub is great."}
    assert find_contamination(pages, items, n=4, min_share=1.0) == {"copy": ["q1"]}


def test_elo_is_zero_sum_and_rewards_the_winner() -> None:
    ratings = elo_ratings([Vote("a", "b", "a")], k=32)
    assert ratings["a"] == pytest.approx(1016.0)
    assert ratings["a"] + ratings["b"] == pytest.approx(2000.0)


def test_bradley_terry_ignores_vote_order_but_elo_does_not() -> None:
    votes = simulate_votes(n_votes=200, seed=3)
    bt, bt_rev = bradley_terry_ratings(votes), bradley_terry_ratings(votes[::-1])
    for m in bt:
        assert bt[m] == pytest.approx(bt_rev[m], abs=1e-6)
    elo, elo_rev = elo_ratings(votes), elo_ratings(votes[::-1])
    assert any(abs(elo[m] - elo_rev[m]) > 1 for m in elo)


def test_simulated_votes_are_reproducible() -> None:
    assert simulate_votes(n_votes=50, seed=1) == simulate_votes(n_votes=50, seed=1)


def test_parse_verdict() -> None:
    assert parse_verdict("A") == "A"
    assert parse_verdict("Answer B is better.") == "B"
    assert parse_verdict("Both are fine") is None


def test_judge_swap_test_detects_position_bias() -> None:
    pairs = [{"prompt": "q", "chosen": "short good", "rejected": "short bad"}] * 4
    report = judge_swap_test(pairs, FakeProvider(responder=biased_judge_responder(position_bonus=5)))
    assert report.first_position_rate == 1.0  # always picks whatever is shown first
    assert report.consistency == 0.0


def test_strip_padding_removes_common_sentences() -> None:
    filler = "Thanks for writing."
    pairs = [{"prompt": "q", "chosen": f"Answer {i}. {filler}", "rejected": f"Wrong {i}."} for i in range(20)]
    assert strip_padding(pairs)[0]["chosen"] == "Answer 0."
