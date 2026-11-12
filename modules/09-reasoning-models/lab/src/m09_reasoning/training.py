"""Training-time techniques, simulated: STaR (filter correct rationales, fine-tune) and a tiny GRPO loop.

Nothing here trains a neural network. "Fine-tuning" maps the data you collected to a new
:class:`SimConfig` (higher skill, fewer traps), which is a stand-in for new weights. What *is* real is
the loop around it: what gets sampled, what the verifier keeps, and how advantages are computed.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .problems import FAMILIES, Problem, check_answer
from .reasoner import Attempt, make_reasoner, simulator_of
from .simulator import SimConfig
from .verifiers import first_bad_step

SFT_GAIN = 0.6  # how far one round of fine-tuning on full coverage moves skill towards 1 and traps towards 0


def split_train_test(problems: Sequence[Problem]) -> tuple[list[Problem], list[Problem]]:
    """Odd ids train, even ids test: every family appears on both sides, no problem on both."""
    train = [p for p in problems if int(p.id[1:]) % 2 == 1]
    test = [p for p in problems if int(p.id[1:]) % 2 == 0]
    return train, test


def evaluate(
    config: SimConfig, problems: Sequence[Problem], seed: int = 0
) -> tuple[float, dict[str, float], float]:
    """One CoT sample per problem. Returns (accuracy, accuracy per family, mean completion tokens)."""
    reasoner = make_reasoner(config, seed=seed)
    hits: dict[str, list[bool]] = {}
    tokens = []
    for p in problems:
        attempt = reasoner.solve(p, "cot")
        hits.setdefault(p.type, []).append(check_answer(p, attempt.answer))
        tokens.append(attempt.completion_tokens)
    flat = [h for v in hits.values() for h in v]
    by_family = {t: sum(v) / len(v) for t, v in sorted(hits.items())}
    return sum(flat) / len(flat), by_family, sum(tokens) / len(tokens)


# --------------------------------------------------------------------------- #
# STaR: Self-Taught Reasoner                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rationale:
    problem: Problem
    attempt: Attempt
    source: str  # "solved" (found on its own) or "hint" (rationalisation)

    @property
    def faithful(self) -> bool:
        """Every written calculation checks out (a right answer can still hide a wrong step)."""
        return first_bad_step(self.attempt.steps) is None


@dataclass(frozen=True)
class StarRound:
    round: int
    test_accuracy: float
    by_family: dict[str, float]
    kept_solved: int | None  # None for the final row (evaluation only)
    kept_hint: int | None
    unfaithful: int | None
    generated_tokens: int | None


def finetune(base: SimConfig, kept: Sequence[Rationale], train: Sequence[Problem], name: str) -> SimConfig:
    """Simulated SFT, always starting from the *base* model (as STaR does).

    More good rationales for a family -> higher skill and fewer traps for that family. Each unfaithful
    rationale teaches a wrong step, so we let it cancel one good example.
    """
    size = Counter(p.type for p in train)
    good = Counter(r.problem.type for r in kept if r.faithful)
    bad = Counter(r.problem.type for r in kept if not r.faithful)
    skill, trap = dict(base.skill), dict(base.trap)
    for family in FAMILIES:
        coverage = max(0.0, (good[family] - bad[family]) / size[family]) if size[family] else 0.0
        skill[family] = base.skill[family] + (1 - base.skill[family]) * SFT_GAIN * coverage
        trap[family] = base.trap[family] * (1 - SFT_GAIN * coverage)
    return replace(base, name=name, skill=skill, trap=trap)


def collect_rationales(
    config: SimConfig, train: Sequence[Problem], *, rationalize: bool, strict: bool, seed: int = 1
) -> tuple[list[Rationale], int]:
    """One STaR outer step: sample, keep what reaches the right answer, rationalise the failures."""
    reasoner = make_reasoner(config, seed=seed)
    kept: list[Rationale] = []
    for p in train:
        attempt = reasoner.solve(p, "cot")
        candidate = Rationale(p, attempt, "solved")
        if check_answer(p, attempt.answer) and (candidate.faithful or not strict):
            kept.append(candidate)
            continue
        if rationalize:
            hinted = Rationale(p, reasoner.solve_with_hint(p), "hint")
            if check_answer(p, hinted.attempt.answer) and (hinted.faithful or not strict):
                kept.append(hinted)
    return kept, reasoner.usage.completion_tokens


def star(
    train: Sequence[Problem],
    test: Sequence[Problem],
    *,
    rounds: int = 4,
    rationalize: bool = True,
    strict: bool = False,
    base: SimConfig | None = None,
) -> list[StarRound]:
    base = base or SimConfig()
    config = base
    history: list[StarRound] = []
    for r in range(rounds + 1):
        accuracy, by_family, _ = evaluate(config, test)
        if r == rounds:
            history.append(StarRound(r, accuracy, by_family, None, None, None, None))
            break
        kept, tokens = collect_rationales(config, train, rationalize=rationalize, strict=strict, seed=1 + r)
        history.append(
            StarRound(
                r,
                accuracy,
                by_family,
                kept_solved=sum(k.source == "solved" for k in kept),
                kept_hint=sum(k.source == "hint" for k in kept),
                unfaithful=sum(not k.faithful for k in kept),
                generated_tokens=tokens,
            )
        )
        config = finetune(base, kept, train, name=f"star-{r + 1}")
    return history


# --------------------------------------------------------------------------- #
# RL with verifiable rewards: a GRPO-style toy                                #
# --------------------------------------------------------------------------- #


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """GRPO: compare each sample with its own group. (r - mean) / std; all zeros if every reward is equal."""
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    if std < 1e-9:
        return [0.0] * len(rewards)
    return [(r - mean) / std for r in rewards]


@dataclass(frozen=True)
class RLStep:
    step: int
    test_accuracy: float
    tokens: float  # mean completion tokens per test answer
    check_prob: float
    reread_prob: float
    train_reward: float | None  # None for the final row (evaluation only, no update)
    zero_signal_groups: int | None  # groups where every sample got the same reward (no learning signal)


def policy_config(base: SimConfig, phi_check: float, phi_reread: float) -> SimConfig:
    """The toy policy has two parameters: logits for 'check each step' and 're-read the question'."""
    return replace(base, name="rl", check_prob=sigmoid(phi_check), reread_prob=sigmoid(phi_reread))


def grpo_toy(
    train: Sequence[Problem],
    test: Sequence[Problem],
    *,
    steps: int = 8,
    group: int = 8,
    lr: float = 4.0,
    base: SimConfig | None = None,
) -> list[RLStep]:
    """REINFORCE with group-relative advantages and a 0/1 reward from the checker (no KL, no clipping).

    The reward only says "right" or "wrong". Nothing rewards length or checking directly; the policy
    discovers that checking pays because checks sometimes catch errors.
    """
    base = base or SimConfig()
    start = math.log(0.1 / 0.9)  # the base model checks a step, or re-reads the question, 10% of the time
    phi_check, phi_reread = start, start
    history: list[RLStep] = []
    for step in range(steps + 1):
        config = policy_config(base, phi_check, phi_reread)
        accuracy, _, tokens = evaluate(config, test)
        if step == steps:
            history.append(RLStep(step, accuracy, tokens, config.check_prob, config.reread_prob, None, None))
            break
        reasoner = make_reasoner(config, seed=100 + step)
        sim = simulator_of(reasoner)
        grad_check = grad_reread = 0.0
        count, total_reward, flat = 0, 0.0, 0
        for p in train:
            rewards, traces = [], []
            for _ in range(group):
                attempt = reasoner.solve(p, "cot")
                rewards.append(1.0 if check_answer(p, attempt.answer) else 0.0)
                traces.append(sim.last_trace)
            advantages = group_advantages(rewards)
            flat += all(a == 0.0 for a in advantages)
            total_reward += sum(rewards)
            for adv, trace in zip(advantages, traces, strict=True):
                count += 1
                # Bernoulli(c) decision x: d log P(x) / d logit = x - c
                grad_check += adv * sum(float(did) - c for c, did in trace.check_decisions)
                grad_reread += adv * sum(float(did) - c for c, did in trace.reread_decisions)
        history.append(
            RLStep(step, accuracy, tokens, config.check_prob, config.reread_prob, total_reward / count, flat)
        )
        phi_check += lr * grad_check / count
        phi_reread += lr * grad_reread / count
    return history
