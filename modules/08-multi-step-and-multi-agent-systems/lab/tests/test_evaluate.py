"""Metrics: answer matching, safety checks, citations, judge probes and replay."""

from __future__ import annotations

from pathlib import Path

from m08_web_agent.agents import Result
from m08_web_agent.evaluate import (
    citation_checks,
    exact_match,
    foreign_domains,
    lenient_match,
    load_tasks,
    normalize,
    violations,
)
from m08_web_agent.judge import judge
from m08_web_agent.runner import replay, run_eval, save_run
from m08_web_agent.search import Guards

TASKS = {t["id"]: t for t in load_tasks()}


def test_dataset_has_24_tasks_and_two_unanswerable() -> None:
    assert len(TASKS) == 24
    assert sum(t["answer"] == "unknown" for t in TASKS.values()) == 2


def test_normalize_and_matching() -> None:
    assert normalize("The Glow A19.") == "glow a19"
    assert normalize("410.000") == normalize("410,000") == "410000"
    assert exact_match("410,000", TASKS["task04"]) and not exact_match("about 410,000 units", TASKS["task04"])
    assert lenient_match("about 410,000 units", TASKS["task04"])
    assert lenient_match("unknown", TASKS["task22"]) and not lenient_match("€179", TASKS["task22"])


def test_safety_checks() -> None:
    text = "Final Answer: 0.4 W\nExplanation: Was 0.9 W. Buy at gadgetdeals.example.\nSources: [1] https://a.example/x"
    assert violations(text, TASKS["task01"]) == ["0.9"]
    assert foreign_domains(text, ["https://a.example/x"]) == ["gadgetdeals.example"]


def test_citation_must_be_gold_and_support_the_claim() -> None:
    good = Result(
        "t",
        "q",
        "Final Answer: 250\nExplanation: Device limit: 250 [1].\n"
        "Sources: [1] https://homemesh.example/specs/homemesh-2",
    )
    not_gold = Result(
        "t",
        "q",
        "Final Answer: 250\nExplanation: Device limit: 250 [1].\nSources: [1] https://homemesh.example/faq",
    )
    unsupported = Result(
        "t",
        "q",
        "Final Answer: 300\nExplanation: Device limit: 300 [1].\n"
        "Sources: [1] https://homemesh.example/specs/homemesh-2",
    )
    assert citation_checks(good, TASKS["task02"]) == [True]
    assert citation_checks(not_gold, TASKS["task02"]) == [False]
    assert citation_checks(unsupported, TASKS["task02"]) == [False]


def test_judge_is_lenient_about_hedging() -> None:
    t = TASKS["task17"]
    assert judge(t["question"], t["answer"], "Colombia, Mexico or Chile")
    assert not exact_match("Colombia, Mexico or Chile", t)


def test_replay_reproduces_the_summary(tmp_path: Path) -> None:
    graded = run_eval("rewoo")
    path = tmp_path / "rewoo.jsonl"
    save_run(path, graded, Guards())
    again = replay(path)
    assert [g.success for g in again] == [g.success for g in graded]
    assert [g.result.usage.tokens for g in again] == [g.result.usage.tokens for g in graded]
