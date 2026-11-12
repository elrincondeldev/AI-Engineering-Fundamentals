"""Inference-time scaling: parallel sampling (vote, best-of-N) and sequential revision."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .problems import Problem, check_answer, parse_number
from .reasoner import Attempt, Reasoner
from .verifiers import SimulatedPRM, first_bad_step


def answer_key(answer: str | None) -> str | None:
    """Normalise an answer so that '175.1' and '175.10' vote together."""
    if answer is None:
        return None
    number = parse_number(answer)
    return str(number) if number is not None else answer.strip()


def majority_vote(answers: Sequence[str | None]) -> str | None:
    """Self-consistency: the most frequent answer wins; ties go to the answer seen first."""
    keys = [answer_key(a) for a in answers]
    counts = Counter(k for k in keys if k is not None)
    if not counts:
        return None
    top = max(counts.values())
    for answer, key in zip(answers, keys, strict=True):
        if key is not None and counts[key] == top:
            return answer
    return None


def best_of_n(attempts: Sequence[Attempt], scores: Sequence[float]) -> Attempt:
    """Best-of-N: keep the attempt with the highest verifier score (first one on ties)."""
    best = max(range(len(attempts)), key=lambda i: (scores[i], -i))
    return attempts[best]


def sample(
    reasoner: Reasoner, problems: Sequence[Problem], n: int, mode: str = "cot"
) -> dict[str, list[Attempt]]:
    """Draw ``n`` independent samples per problem (the same prompt, called n times)."""
    return {p.id: [reasoner.solve(p, mode) for _ in range(n)] for p in problems}


@dataclass(frozen=True)
class VoteStats:
    """Why voting stops improving: where the plurality answer at N comes from."""

    problems: int
    plurality_wrong: int  # problems whose most frequent answer is wrong
    wrong_from_trap: int  # ...of those, how many have a systematic (trap) answer as plurality
    never_right: int  # problems with no correct sample at all


def vote_stats(
    problems: Sequence[Problem],
    samples: dict[str, list[Attempt]],
    n: int,
    is_trap: Callable[[Problem, Attempt], bool],
) -> VoteStats:
    wrong = trap = never = 0
    for p in problems:
        attempts = samples[p.id][:n]
        winner = majority_vote([a.answer for a in attempts])
        if not check_answer(p, winner):
            wrong += 1
            first = next(a for a in attempts if answer_key(a.answer) == answer_key(winner))
            trap += is_trap(p, first)
        never += not any(check_answer(p, a.answer) for a in attempts)
    return VoteStats(len(problems), wrong, trap, never)


# --------------------------------------------------------------------------- #
# Sequential revision                                                         #
# --------------------------------------------------------------------------- #

Feedback = Callable[[Problem, Attempt], str | None]  # None means "looks fine, stop"


def self_feedback(reasoner: Reasoner) -> Feedback:
    """The model critiques its own work (no external verifier)."""

    def feedback(problem: Problem, attempt: Attempt) -> str | None:
        verdict = reasoner.critique(problem, attempt)
        return verdict if verdict.startswith("Step") else None

    return feedback


def recheck_feedback(problem: Problem, attempt: Attempt) -> str | None:
    """Exact arithmetic re-check: points at the first step whose numbers do not add up."""
    k = first_bad_step(attempt.steps)
    return None if k is None else f"Step {k} is wrong: its arithmetic does not check out."


def prm_feedback(prm: SimulatedPRM) -> Feedback:
    """A process reward model points at the first low-scoring step."""

    def feedback(problem: Problem, attempt: Attempt) -> str | None:
        k = prm.first_bad_step(problem, attempt.steps)
        return None if k is None else f"Step {k} is wrong: a process reward model gave it a low score."

    return feedback


def revise_loop(
    reasoner: Reasoner, problem: Problem, first: Attempt, feedback: Feedback, rounds: int = 3
) -> list[Attempt]:
    """Attempt, get feedback, revise (only when the feedback flags something). One attempt per round."""
    history = [first]
    current = first
    for _ in range(rounds):
        note = feedback(problem, current)
        if note is not None:
            current = reasoner.revise(problem, current, note)
        history.append(current)
    return history


def accuracy(problems: Sequence[Problem], answers: Sequence[str | None]) -> float:
    return sum(check_answer(p, a) for p, a in zip(problems, answers, strict=True)) / len(problems)


def mean(values: Sequence[float] | Sequence[int]) -> float:
    return float(sum(values)) / len(values) if values else 0.0
