"""Reward models (Bradley–Terry), the length-bias trap, reward hacking, DPO and the PPO KL penalty.

A reward model gives a score r(prompt, response). The Bradley–Terry model says the probability
that people prefer response A over response B is sigmoid(r(A) - r(B)). Fitting it to preference
pairs is logistic regression on the difference of the two feature vectors.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .data import read_jsonl
from .text import tokenize


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def load_pairs(split: str) -> list[dict[str, str]]:
    """Preference pairs of one split (``train`` or ``test``) from ``preferences/pairs.jsonl``."""
    return [row for row in read_jsonl("preferences/pairs.jsonl") if row["split"] == split]


def words(text: str) -> list[str]:
    return [t for t in tokenize(text) if any(ch.isalnum() for ch in t)]


# --------------------------------------------------------------------------- #
# Verifiable rewards                                                          #
# --------------------------------------------------------------------------- #

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def verifiable_reward(model_answer: str, gold: str, tol: float = 0.005) -> float:
    """1.0 if the last number in the model's answer equals the gold number, else 0.0.

    No reward model is needed: a program checks the answer. This is the idea behind
    reinforcement learning with verifiable rewards (maths, code with tests, ...).
    """
    found = _NUMBER_RE.findall(model_answer)
    if not found:
        return 0.0
    value = float(found[-1].replace(",", "."))
    return 1.0 if abs(value - float(gold)) <= tol else 0.0


# --------------------------------------------------------------------------- #
# Features                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Featurizer:
    """Turns a response into a feature vector: optional length, optional bag of words."""

    vocab: tuple[str, ...] = ()
    use_length: bool = True

    @classmethod
    def build(cls, pairs: Sequence[dict[str, str]], use_words: bool, use_length: bool) -> Featurizer:
        vocab: tuple[str, ...] = ()
        if use_words:
            counts = Counter(w for p in pairs for side in ("chosen", "rejected") for w in set(words(p[side])))
            vocab = tuple(sorted(w for w, c in counts.items() if c >= 2))
        return cls(vocab=vocab, use_length=use_length)

    @property
    def names(self) -> list[str]:
        return (["<length>"] if self.use_length else []) + list(self.vocab)

    def __call__(self, text: str) -> np.ndarray:
        toks = words(text)
        feats = []
        if self.use_length:
            feats.append(math.log1p(len(toks)))
        if self.vocab:
            present = set(toks)
            feats.extend(1.0 if w in present else 0.0 for w in self.vocab)
        return np.array(feats, dtype=np.float64)

    def matrix(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self(t) for t in texts])


# --------------------------------------------------------------------------- #
# Bradley–Terry reward model                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class RewardModel:
    """A linear reward r(response) = w . features(response) fitted with the Bradley–Terry loss."""

    featurizer: Featurizer
    weights: np.ndarray

    def score(self, text: str) -> float:
        return float(self.featurizer(text) @ self.weights)

    def top_weights(self, n: int = 5) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """The ``n`` most positive and most negative features."""
        order = np.argsort(self.weights, kind="stable")
        names = self.featurizer.names
        pos = [(names[i], float(self.weights[i])) for i in order[::-1][:n]]
        neg = [(names[i], float(self.weights[i])) for i in order[:n]]
        return pos, neg


def bradley_terry_loss(w: np.ndarray, diff: np.ndarray, l2: float) -> float:
    """Mean of -log sigmoid(r(chosen) - r(rejected)) plus an L2 penalty."""
    margins = diff @ w
    return float(np.mean(np.logaddexp(0.0, -margins)) + 0.5 * l2 * w @ w)


def fit_reward_model(
    pairs: Sequence[dict[str, str]],
    featurizer: Featurizer,
    l2: float = 0.01,
    lr: float = 0.5,
    steps: int = 500,
) -> RewardModel:
    """Full-batch gradient descent on the Bradley–Terry loss. Deterministic: weights start at 0."""
    xc = featurizer.matrix([p["chosen"] for p in pairs])
    xr = featurizer.matrix([p["rejected"] for p in pairs])
    diff = xc - xr
    w = np.zeros(diff.shape[1])
    for _ in range(steps):
        margins = diff @ w
        grad = -(diff * sigmoid(-margins)[:, None]).mean(axis=0) + l2 * w
        w -= lr * grad
    return RewardModel(featurizer, w)


def pairwise_accuracy(model: RewardModel, pairs: Sequence[dict[str, str]]) -> float:
    """Share of pairs where the chosen response gets the higher reward (ties count half)."""
    total = 0.0
    for p in pairs:
        rc, rr = model.score(p["chosen"]), model.score(p["rejected"])
        total += 1.0 if rc > rr else 0.5 if rc == rr else 0.0
    return total / len(pairs)


def chosen_is_longer(pairs: Sequence[dict[str, str]]) -> float:
    """The "longer answer wins" baseline: share of pairs where the chosen response is longer."""
    return sum(len(p["chosen"]) > len(p["rejected"]) for p in pairs) / len(pairs)


def length_balanced(pairs: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Drop pairs so that "chosen is longer" and "chosen is shorter" are equally common.

    Keeps every pair of the minority kind and the first pairs (in file order) of the majority kind.
    """
    longer = [p for p in pairs if len(p["chosen"]) > len(p["rejected"])]
    shorter = [p for p in pairs if len(p["chosen"]) <= len(p["rejected"])]
    n = min(len(longer), len(shorter))
    keep = {p["id"] for p in longer[:n] + shorter[:n]}
    return [p for p in pairs if p["id"] in keep]


def most_common_sentence(pairs: Sequence[dict[str, str]], side: str = "chosen") -> str:
    """The sentence that appears in the most responses: in this data, polite filler."""
    counts: Counter[str] = Counter()
    for p in pairs:
        sentences = {s.strip() + "." for s in p[side].split(". ") if s.strip()}
        counts.update(sentences)
    sentence = counts.most_common(1)[0][0]
    return sentence[:-1] if sentence.endswith("..") else sentence


def hack_rate(model: RewardModel, pairs: Sequence[dict[str, str]], padding: str) -> float:
    """Share of pairs where padding the *rejected* answer lifts it above the chosen one.

    A policy optimised against this reward model would learn to add the padding: reward hacking.
    """
    wins = sum(model.score(f"{p['rejected']} {padding}") > model.score(p["chosen"]) for p in pairs)
    return wins / len(pairs)


def prompt_disjoint(
    train: Sequence[dict[str, str]], test: Sequence[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split by prompt so no test question was seen in training.

    Distinct prompts are listed in order of first appearance; even positions go to training
    (train-split pairs only) and odd positions to testing (test-split pairs only).
    """
    prompts: list[str] = []
    for p in train:
        if p["prompt"] not in prompts:
            prompts.append(p["prompt"])
    seen = set(prompts[::2])
    return [p for p in train if p["prompt"] in seen], [p for p in test if p["prompt"] not in seen]


@dataclass(frozen=True)
class RewardExperiment:
    """Accuracy (and hack rate) of one reward-model variant."""

    name: str
    n_train_pairs: int
    n_features: int
    train_acc: float
    test_acc: float
    hack_rate: float


def _run(name: str, data: Sequence[dict[str, str]], use_words: bool, test: Sequence[dict[str, str]],
         padding: str) -> tuple[RewardExperiment, RewardModel]:
    feats = Featurizer.build(data, use_words=use_words, use_length=True)
    model = fit_reward_model(data, feats)
    row = RewardExperiment(name, len(data), len(feats.names), pairwise_accuracy(model, data),
                           pairwise_accuracy(model, test), hack_rate(model, test, padding))
    return row, model


def reward_experiments() -> dict[str, list[RewardExperiment]]:
    """The two tables of lesson 2.3.

    ``official``: the dataset's own train/test split (the same prompts appear in both).
    ``new_prompts``: train and test on disjoint prompts, with and without length-balanced data.
    """
    train, test = load_pairs("train"), load_pairs("test")
    padding = most_common_sentence(train)
    official = [
        _run("length only", train, False, test, padding)[0],
        _run("words + length", train, True, test, padding)[0],
    ]
    train_a, test_b = prompt_disjoint(train, test)
    new_prompts = [
        _run("length only", train_a, False, test_b, padding)[0],
        _run("words + length", train_a, True, test_b, padding)[0],
        _run("words + length, balanced", length_balanced(train_a), True, test_b, padding)[0],
    ]
    return {"official": official, "new_prompts": new_prompts}


# --------------------------------------------------------------------------- #
# DPO and the PPO KL penalty on toy log-probabilities                         #
# --------------------------------------------------------------------------- #


def dpo_loss(
    policy_chosen: float, policy_rejected: float, ref_chosen: float, ref_rejected: float, beta: float = 0.1
) -> float:
    """Direct Preference Optimization loss for one pair, from sequence log-probabilities.

    loss = -log sigmoid(beta * [(policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)])
    """
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return float(np.logaddexp(0.0, -beta * margin))  # = -log sigmoid(beta * margin)


def implicit_reward(policy_logp: float, ref_logp: float, beta: float = 0.1) -> float:
    """DPO's implicit reward of a response: beta * log(pi(y|x) / pi_ref(y|x))."""
    return beta * (policy_logp - ref_logp)


def kl_penalized_reward(reward: float, policy_logp: float, ref_logp: float, beta: float = 0.1) -> float:
    """The per-sample reward PPO-based RLHF maximises: r - beta * (log pi - log pi_ref)."""
    return reward - beta * (policy_logp - ref_logp)


# Toy sequence log-probabilities (natural log) for one preference pair. The reference model
# gives log p(chosen) = -20 and log p(rejected) = -18: it slightly prefers the rejected answer.
REF_CHOSEN, REF_REJECTED = -20.0, -18.0
DPO_SCENARIOS: dict[str, tuple[float, float]] = {
    "policy = reference (start of training)": (-20.0, -18.0),
    "chosen up 1, rejected down 1": (-19.0, -19.0),
    "chosen down 2, rejected down 5": (-22.0, -23.0),
    "chosen down 1, rejected up 1 (wrong way)": (-21.0, -17.0),
}


def dpo_table(betas: Sequence[float] = (0.1, 0.5)) -> list[tuple[str, float, list[float]]]:
    """(scenario, log-ratio margin, DPO loss for each beta) for the toy scenarios."""
    out = []
    for name, (pc, pr) in DPO_SCENARIOS.items():
        margin = (pc - REF_CHOSEN) - (pr - REF_REJECTED)
        out.append((name, margin, [dpo_loss(pc, pr, REF_CHOSEN, REF_REJECTED, b) for b in betas]))
    return out
