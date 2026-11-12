"""The 160 Hearthwise reasoning problems and their exact (rule-based) verifiers."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from .data import read_jsonl

FAMILIES = ("order_total", "energy", "hub_capacity", "target")
NUMERIC_FAMILIES = ("order_total", "energy", "hub_capacity")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Problem:
    """One row of ``datasets/reasoning/problems.jsonl``."""

    id: str
    type: str
    difficulty: int
    question: str
    answer: str
    answer_type: str
    steps: tuple[str, ...]
    numbers: tuple[int, ...] = ()
    target: int | None = None

    @property
    def is_numeric(self) -> bool:
        return self.answer_type == "number"


@lru_cache(maxsize=1)
def load_problems() -> tuple[Problem, ...]:
    """Read every problem once (the file is never modified)."""
    rows = read_jsonl("reasoning/problems.jsonl")
    return tuple(
        Problem(
            id=r["id"],
            type=r["type"],
            difficulty=int(r["difficulty"]),
            question=r["question"],
            answer=r["answer"],
            answer_type=r["answer_type"],
            steps=tuple(r["steps"]),
            numbers=tuple(r.get("numbers", ())),
            target=r.get("target"),
        )
        for r in rows
    )


def numeric_problems() -> list[Problem]:
    return [p for p in load_problems() if p.is_numeric]


def target_problems() -> list[Problem]:
    return [p for p in load_problems() if p.type == "target"]


def fmt(value: Fraction | int) -> str:
    """The dataset's number format: integers without decimals, otherwise two decimals."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.2f}"


def parse_number(text: str | None) -> Fraction | None:
    """First number in ``text`` as an exact fraction, or None."""
    if not text:
        return None
    match = _NUMBER_RE.search(text.replace(",", ""))
    return Fraction(match.group(0)) if match else None


_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def evaluate_expression(expr: str) -> tuple[Fraction, list[int]]:
    """Evaluate an expression made only of integers, + - * / and brackets. Returns (value, numbers used)."""
    used: list[int] = []

    def walk(node: ast.AST) -> Fraction:
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("division by zero")
            return _OPS[type(node.op)](left, right)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            used.append(node.value)
            return Fraction(node.value)
        raise ValueError(f"not allowed in a puzzle answer: {ast.dump(node)[:60]}")

    tree = ast.parse(expr.strip(), mode="eval")
    return walk(tree.body), used


def verify_expression(numbers: tuple[int, ...] | list[int], target: int, expr: str | None) -> bool:
    """Exact verifier for ``target`` puzzles: each number used once and the value equals the target."""
    if not expr:
        return False
    try:
        value, used = evaluate_expression(expr)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return False
    return sorted(used) == sorted(numbers) and value == target


def check_answer(problem: Problem, answer: str | None) -> bool:
    """Outcome check against the reference: exact for numbers, the verifier for expressions."""
    if problem.answer_type == "expression":
        assert problem.target is not None
        return verify_expression(problem.numbers, problem.target, answer)
    got = parse_number(answer)
    return got is not None and abs(got - Fraction(problem.answer)) < Fraction(1, 1000)
