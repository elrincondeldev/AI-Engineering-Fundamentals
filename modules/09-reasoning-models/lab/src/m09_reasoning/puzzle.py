"""Helpers for the ``target`` puzzles: legal moves, exact numbers and an exhaustive solvability check."""

from __future__ import annotations

import itertools
import re
from fractions import Fraction
from functools import cache

Item = tuple[Fraction, str]  # (value the solver believes, expression that produced it)
Move = tuple[int, str, int]  # (index of a, operator, index of b)
_MOVE_RE = re.compile(r"^\s*(-?[\d/]+)\s*([-+*/])\s*(-?[\d/]+)\s*=\s*(-?[\d/.]+)")


def apply(op: str, a: Fraction, b: Fraction) -> Fraction | None:
    """Apply one operator; None for division by zero."""
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b if b != 0 else None
    raise ValueError(op)


def exact(value: Fraction) -> str:
    """Write a number without rounding: 8, 8/3."""
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def legal_moves(values: list[Fraction]) -> list[Move]:
    """Every move with a non-negative result (the rule used to build the dataset)."""
    moves: list[Move] = []
    for i, j in itertools.permutations(range(len(values)), 2):
        for op in "+-*/":
            if op in "+*" and i > j:
                continue  # commutative: keep one order
            out = apply(op, values[i], values[j])
            if out is not None and out >= 0:
                moves.append((i, op, j))
    return moves


def combine(items: list[Item], move: Move, written: Fraction | None = None) -> list[Item]:
    """Replace the two operands by the result. ``written`` is the value the solver *claims*."""
    i, op, j = move
    (a, ea), (b, eb) = items[i], items[j]
    result = apply(op, a, b)
    assert result is not None
    value = result if written is None else written
    rest = [it for k, it in enumerate(items) if k not in (i, j)]
    return rest + [(value, f"({ea} {op} {eb})")]


def strip_outer(expr: str) -> str:
    """Remove one pair of redundant outer brackets."""
    if not (expr.startswith("(") and expr.endswith(")")):
        return expr
    depth = 0
    for k, ch in enumerate(expr):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and k < len(expr) - 1:
            return expr
    return expr[1:-1]


def parse_move_line(line: str) -> tuple[Fraction, str, Fraction, Fraction] | None:
    """Parse 'a op b = c' (exact numbers); None if the line is not a move."""
    m = _MOVE_RE.match(line)
    if not m:
        return None
    try:
        return Fraction(m.group(1)), m.group(2), Fraction(m.group(3)), Fraction(m.group(4))
    except (ValueError, ZeroDivisionError):
        return None


def find_move(items: list[Item], a: Fraction, op: str, b: Fraction) -> Move | None:
    """Indices of two different items with values ``a`` and ``b``."""
    for i, (va, _) in enumerate(items):
        if va != a:
            continue
        for j, (vb, _) in enumerate(items):
            if j != i and vb == b:
                return (i, op, j)
    return None


@cache
def _reachable(values: tuple[Fraction, ...], target: Fraction) -> bool:
    if len(values) == 1:
        return values[0] == target
    vals = list(values)
    for i, op, j in legal_moves(vals):
        out = apply(op, vals[i], vals[j])
        assert out is not None
        rest = [v for k, v in enumerate(vals) if k not in (i, j)] + [out]
        if _reachable(tuple(sorted(rest)), target):
            return True
    return False


def reachable(values: list[Fraction], target: Fraction | int) -> bool:
    """Exhaustive check: can these numbers still reach the target?"""
    return _reachable(tuple(sorted(values)), Fraction(target))
