"""Checks for exercises.py. Each test is skipped until you implement the function."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from m09_reasoning import exercises as ex


def _call(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return fn(*args)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} is not implemented yet (src/m09_reasoning/exercises.py)")


def test_pass_at_k() -> None:
    assert _call(ex.pass_at_k, 10, 0, 3) == 0.0
    assert _call(ex.pass_at_k, 10, 10, 1) == 1.0
    assert math.isclose(_call(ex.pass_at_k, 5, 1, 1), 0.2)
    assert math.isclose(_call(ex.pass_at_k, 5, 1, 5), 1.0)
    assert math.isclose(_call(ex.pass_at_k, 4, 2, 2), 1 - 1 / 6)
    assert math.isclose(_call(ex.pass_at_k, 15, 6, 1), 6 / 15)  # pass@1 = fraction of correct samples


def test_weighted_vote() -> None:
    assert _call(ex.weighted_vote, ["52", "22", "22"], [0.9, 0.3, 0.3]) == "52"
    assert _call(ex.weighted_vote, ["175.1", "175.10", "90"], [0.4, 0.4, 0.7]) == "175.1"
    assert _call(ex.weighted_vote, [None, "3"], [5.0, 0.1]) == "3"
    assert _call(ex.weighted_vote, ["a", "b"], [0.5, 0.5]) == "a"
    assert _call(ex.weighted_vote, [None], [1.0]) is None


def test_adaptive_vote() -> None:
    stream = iter(["7", "7", "7", "9", "9"])
    assert _call(ex.adaptive_vote, lambda: next(stream), 5, 3) == ("7", 3)
    stream = iter(["1", "2", "1", "2", "1"])
    assert _call(ex.adaptive_vote, lambda: next(stream), 5, 3) == ("1", 5)
    stream = iter(["4.0", "4", "4.00"])
    assert _call(ex.adaptive_vote, lambda: next(stream), 9, 3) == ("4.0", 3)
