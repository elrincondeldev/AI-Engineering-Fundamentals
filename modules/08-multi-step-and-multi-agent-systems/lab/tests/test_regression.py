"""Regression suite: the exact set of tasks each agent fails, with and without guards.

If a change to the code moves a task in or out of these sets, this test fails and
forces you to look. Update the expectations only after you understand why.
"""

from __future__ import annotations

import pytest

from m08_web_agent.evaluate import summarize
from m08_web_agent.runner import run_eval
from m08_web_agent.search import Guards

EXPECTED_FAILURES = {
    ("react", "guarded"): (),
    ("react", "naive"): ("task04", "task19"),
    ("rewoo", "guarded"): (),
    ("rewoo", "naive"): ("task04", "task19"),
    ("plan", "guarded"): (),
    ("plan", "naive"): ("task04", "task19"),
    ("reflexion", "guarded"): (),
    ("reflexion", "naive"): (),
    ("tree", "guarded"): (),
    ("tree", "naive"): ("task02", "task19"),
    ("mcts", "guarded"): (),
    ("mcts", "naive"): ("task02", "task19"),
    ("multi", "guarded"): (),
    ("multi", "naive"): ("task02",),
}


@pytest.mark.parametrize(("agent", "harness"), sorted(EXPECTED_FAILURES))
def test_failures_do_not_move(agent: str, harness: str) -> None:
    guards = Guards() if harness == "guarded" else Guards.none()
    assert summarize(run_eval(agent, guards)).failed_ids == EXPECTED_FAILURES[(agent, harness)]


def test_runs_are_deterministic() -> None:
    a = [g.result.answer_text for g in run_eval("mcts", Guards.none())]
    b = [g.result.answer_text for g in run_eval("mcts", Guards.none())]
    assert a == b
