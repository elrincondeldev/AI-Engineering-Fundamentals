"""Verifiers: an exact arithmetic re-check, and *simulated* outcome and process reward models.

- ``first_bad_step`` is a real, rule-based verifier: it recomputes every calculation written in a step.
  It catches slips, but not misreadings (counting bulbs is perfectly good arithmetic).
- ``SimulatedORM`` and ``SimulatedPRM`` stand in for trained reward models. They peek at the dataset's
  reference answer or steps and add seeded noise, so they are imperfect in a controlled way. A real ORM
  or PRM is a neural network trained on labelled solutions; it has no access to the reference.
"""

from __future__ import annotations

import ast
import math
import random
import re
from dataclasses import dataclass
from fractions import Fraction

from .problems import Problem, check_answer
from .reasoner import Attempt

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?%?|(?<![A-Za-z])x(?![A-Za-z])|[-+*/()]")
_CLAIM_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")
_COMPARE_RE = re.compile(r"(-?\d+(?:\.\d+)?) is (not )?over €(\d+(?:\.\d+)?)")
_ROUND_RE = re.compile(r"(-?\d+(?:\.\d+)?) years, rounded to (-?\d+(?:\.\d+)?)")


def _exact_eval(expr: str) -> Fraction:
    def walk(node: ast.AST) -> Fraction:
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(repr(node.value))
        raise ValueError("cannot check this expression")

    return walk(ast.parse(expr, mode="eval").body)


def step_is_consistent(step: str) -> bool:
    """True when every calculation or comparison written in ``step`` is right (or there is none)."""
    cmp = _COMPARE_RE.search(step)
    if cmp and (Fraction(cmp.group(1)) > Fraction(cmp.group(3))) == bool(cmp.group(2)):
        return False
    rounding = _ROUND_RE.search(step)
    if rounding and round(float(rounding.group(1)), 1) != float(rounding.group(2)):
        return False
    if step.count("=") != 1:
        return True
    left, right = step.split("=")
    claim = _CLAIM_RE.match(right)
    tokens = _TOKEN_RE.findall(left.split(":")[-1])
    if not claim or not tokens:
        return True
    expr = " ".join("*" if t == "x" else (f"({t[:-1]} / 100)" if t.endswith("%") else t) for t in tokens)
    try:
        value = _exact_eval(expr)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return True
    text = claim.group(1)
    decimals = len(text.split(".")[1]) if "." in text else 0
    tolerance = Fraction(1, 2 * 10**decimals) if decimals else Fraction(0)
    return abs(value - Fraction(text)) <= tolerance


def first_bad_step(steps: tuple[str, ...] | list[str]) -> int | None:
    """1-based index of the first step whose arithmetic does not check out, or None."""
    for k, step in enumerate(steps, 1):
        if not step_is_consistent(step):
            return k
    return None


def _noise(*parts: str) -> float:
    return random.Random("|".join(parts)).gauss(0.0, 1.0)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class SimulatedORM:
    """Outcome reward model: one score per *complete* solution.

    A tidy wrong solution (consistent arithmetic, wrong reading of the question) fools it more than a
    messy one. ``length_bias`` > 0 makes it prefer long answers: the hook for the reward-hacking demo.
    """

    noise: float = 1.0
    length_bias: float = 0.0
    seed: int = 0

    def score(self, problem: Problem, attempt: Attempt) -> float:
        if check_answer(problem, attempt.answer):
            logit = 1.0
        elif first_bad_step(attempt.steps) is None:
            logit = 0.3  # wrong but tidy: hard to tell apart from a right answer
        else:
            logit = -1.0
        logit += self.noise * _noise("orm", str(self.seed), problem.id, attempt.text)
        logit += self.length_bias * attempt.completion_tokens / 100
        return sigmoid(logit)


@dataclass(frozen=True)
class SimulatedPRM:
    """Process reward model: one score per *step* (probability that the step is right).

    Simulated by comparing each step with the reference step at the same position, plus noise.
    Only defined for the numeric families (a puzzle has many valid paths).
    """

    noise: float = 2.0
    seed: int = 0

    def step_scores(self, problem: Problem, steps: tuple[str, ...] | list[str]) -> list[float]:
        scores = []
        for k, step in enumerate(steps):
            match = k < len(problem.steps) and step == problem.steps[k]
            logit = 2.5 if match else -2.0
            logit += self.noise * _noise("prm", str(self.seed), problem.id, str(k), step)
            scores.append(sigmoid(logit))
        return scores

    def score(self, problem: Problem, steps: tuple[str, ...] | list[str]) -> float:
        """Aggregate with the minimum: a chain is only as good as its weakest step."""
        scores = self.step_scores(problem, steps)
        return min(scores) if scores else 0.0

    def first_bad_step(
        self, problem: Problem, steps: tuple[str, ...] | list[str], threshold: float = 0.5
    ) -> int | None:
        for k, s in enumerate(self.step_scores(problem, steps), 1):
            if s < threshold:
                return k
        return None
