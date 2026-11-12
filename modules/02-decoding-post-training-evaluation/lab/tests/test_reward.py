"""Unit tests for the Bradley–Terry reward model, DPO and the KL-penalised reward."""

from __future__ import annotations

import math

import pytest

from m02_post_training.reward import (
    Featurizer,
    dpo_loss,
    fit_reward_model,
    hack_rate,
    implicit_reward,
    kl_penalized_reward,
    length_balanced,
    load_pairs,
    pairwise_accuracy,
    prompt_disjoint,
    verifiable_reward,
)


def pair(pid: str, chosen: str, rejected: str, prompt: str = "q") -> dict[str, str]:
    return {"id": pid, "prompt": prompt, "chosen": chosen, "rejected": rejected}


def test_reward_model_learns_a_content_word() -> None:
    pairs = [pair(str(i), "yes it works", "no it fails") for i in range(10)]
    feats = Featurizer.build(pairs, use_words=True, use_length=False)
    model = fit_reward_model(pairs, feats, steps=200)
    assert pairwise_accuracy(model, pairs) == 1.0
    assert model.score("it works") > model.score("it fails")


def test_length_only_model_prefers_longer_answers() -> None:
    pairs = [pair(str(i), "a long and detailed answer", "short") for i in range(5)]
    model = fit_reward_model(pairs, Featurizer.build(pairs, use_words=False, use_length=True))
    assert model.weights[0] > 0
    assert hack_rate(model, pairs, padding="lots of extra words here") == 1.0


def test_length_balanced_has_as_many_longer_as_shorter() -> None:
    train = load_pairs("train")
    balanced = length_balanced(train)
    longer = sum(len(p["chosen"]) > len(p["rejected"]) for p in balanced)
    assert longer * 2 == len(balanced)


def test_prompt_disjoint_split_has_no_shared_prompts() -> None:
    train_a, test_b = prompt_disjoint(load_pairs("train"), load_pairs("test"))
    assert not {p["prompt"] for p in train_a} & {p["prompt"] for p in test_b}


def test_dpo_loss_properties() -> None:
    assert dpo_loss(-5, -5, -5, -5) == pytest.approx(math.log(2))
    better = dpo_loss(-4, -6, -5, -5, beta=0.1)
    worse = dpo_loss(-6, -4, -5, -5, beta=0.1)
    assert better < math.log(2) < worse
    # Only the difference of log-ratios matters, not the absolute log-probabilities.
    assert dpo_loss(-14, -16, -15, -15) == pytest.approx(better)


def test_implicit_and_kl_penalised_rewards() -> None:
    assert implicit_reward(-10.0, -12.0, beta=0.1) == pytest.approx(0.2)
    assert kl_penalized_reward(1.0, -10.0, -10.0, beta=0.5) == 1.0
    assert kl_penalized_reward(1.0, -8.0, -10.0, beta=0.5) == pytest.approx(0.0)


def test_verifiable_reward() -> None:
    assert verifiable_reward("so the answer is 447", "447") == 1.0
    assert verifiable_reward("Total: 175,1 euros", "175.10") == 1.0
    assert verifiable_reward("I am not sure", "447") == 0.0
    assert verifiable_reward("446", "447") == 0.0
    assert verifiable_reward("175.10 euros (180.05 if you choose express)", "175.10") == 0.0  # lesson 2.3
