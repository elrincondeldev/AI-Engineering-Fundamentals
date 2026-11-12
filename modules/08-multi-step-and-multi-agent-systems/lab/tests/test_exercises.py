"""Tests for your solutions in ``exercises.py``. Each test is skipped until you implement it."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from m08_web_agent import exercises as ex
from m08_web_agent.agents import Result, Step, Usage
from m08_web_agent.skills import Fact
from m08_web_agent.web import load_snapshot


def attempt(fn: Callable, *args: object) -> object:
    try:
        return fn(*args)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} is not implemented yet")


def test_mask_street_addresses() -> None:
    page = load_snapshot().get("https://forum.homeautomators.example/t/holiday-lights-automation")
    out = attempt(ex.mask_street_addresses, page.text)
    assert "Calle Falsa 123" not in out and "[address]" in out
    assert "Plug plus sunset trigger." in out
    assert attempt(ex.mask_street_addresses, "Fix the channel to 6.") == "Fix the channel to 6."


def test_repeated_action() -> None:
    a, b = Step("t", "search_web", "ember", "o"), Step("t", "open_page", "https://x.example/", "o")
    assert attempt(ex.repeated_action, [a, b, b]) is True
    assert attempt(ex.repeated_action, [a, b]) is False
    assert attempt(ex.repeated_action, []) is False


def test_vote_by_domain() -> None:
    spec = Fact("s", "250", "https://homemesh.example/specs/homemesh-2", "2024-04-16", "", "", 0)
    forum = Fact("s", "200", "https://forum.homeautomators.example/t/homemesh-device-limits", "", "", "", 1)
    copy = Fact(
        "s",
        "200",
        "https://forum.homeautomators.example/t/homemesh-device-limits?utm_source=x",
        "",
        "",
        "",
        2,
    )
    assert attempt(ex.vote_by_domain, [spec, forum, copy]).value == "250"
    assert attempt(ex.vote_by_domain, []) is None


def test_citation_recall() -> None:
    r = Result("t", "q", "Final Answer: x\nSources: [1] https://a.example/p?utm_source=n")
    assert attempt(ex.citation_recall, r, ["https://a.example/p", "https://b.example/q"]) == 0.5
    assert attempt(ex.citation_recall, r, []) == 1.0


def test_cost_per_success() -> None:
    runs = [Result("t", "q", "", usage=Usage(1, 90, 10)), Result("t", "q", "", usage=Usage(1, 290, 10))]
    assert attempt(ex.cost_per_success, runs, [True, False]) == 400.0
    assert attempt(ex.cost_per_success, runs, [False, False]) == float("inf")
