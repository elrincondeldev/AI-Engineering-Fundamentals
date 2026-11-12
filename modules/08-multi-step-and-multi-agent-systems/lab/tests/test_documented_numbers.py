"""Every number quoted in the module's Markdown, asserted against a real run.

If one of these fails after a code change, update the lesson text *and* this test
together. Floats are compared after the same rounding the lessons print.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from m08_web_agent.agents import MAX_STEPS
from m08_web_agent.evaluate import Summary, load_tasks, summarize
from m08_web_agent.judge import agreement, run_probes
from m08_web_agent.multi import MultiAgentSystem
from m08_web_agent.reflexion import run_reflexion
from m08_web_agent.runner import replay, run_agent, run_eval, save_run
from m08_web_agent.sanitize import sanitize
from m08_web_agent.search import Guards, SearchEngine
from m08_web_agent.tools import MAX_PAGE_CHARS, WebTools
from m08_web_agent.tree import TreeSearch, run_tree
from m08_web_agent.web import load_snapshot

TASKS = {t["id"]: t for t in load_tasks()}
ON, OFF = Guards(), Guards.none()


@lru_cache(maxsize=None)
def summary(agent: str, guards: Guards = ON, **options: object) -> Summary:
    return summarize(run_eval(agent, guards, **options))


@lru_cache(maxsize=None)
def graded(agent: str, guards: Guards = ON, **options: object):
    return run_eval(agent, guards, **options)


def task_run(agent: str, task_id: str, guards: Guards = ON, **options: object):
    return run_agent(agent, TASKS[task_id]["question"], guards, **options)


def row(s: Summary) -> tuple:
    return (round(s.success, 3), round(s.avg_model_calls, 2), round(s.avg_tokens), round(s.avg_steps, 2))


# --------------------------------------------------------------------------- data and tools


def test_snapshot_sizes() -> None:
    snap = load_snapshot()
    assert (len(snap.pages), len(snap.readable)) == (134, 125)
    assert len(SearchEngine(snap, ON).pages) == 100
    assert len(TASKS) == 24
    assert (MAX_STEPS, MAX_PAGE_CHARS) == (12, 1500)


def test_sanitiser_on_the_forum_page() -> None:
    clean = sanitize(load_snapshot().get("https://forum.homeautomators.example/t/ember-t3-wifi-drops").text)
    assert (clean.removed, clean.masked) == (1, 2)


def test_homemesh_search_top_result() -> None:
    hit = SearchEngine(load_snapshot(), ON).search("HomeMesh 2.0 devices")[0]
    assert (hit.url, hit.tier) == ("https://homemesh.example/specs/homemesh-2", "high")


# --------------------------------------------------------------------------- lesson 8.1


def test_lesson1_task06_trace() -> None:
    r = task_run("react", "task06")
    assert r.short == "30"
    assert (r.usage.calls, r.tool_calls, r.usage.tokens) == (7, 6, 7012)
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (6669, 343)
    assert r.prompt_sizes == [239, 552, 770, 917, 1214, 1406, 1571]
    assert r.prompt_sizes[-1] / r.prompt_sizes[0] > 6
    assert r.sources == [
        "https://smarthomeweekly.example/reviews/corvid-thermo-x-review-updated",
        "https://smarthomeweekly.example/reviews/ember-t3-review",
    ]


def test_lesson1_budget() -> None:
    r = task_run("react", "task11", max_steps=3)
    assert r.short == "unknown" and r.stopped == "budget"
    assert (r.usage.calls, r.tool_calls, r.usage.tokens) == (4, 3, 2709)
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (2527, 182)


def test_lesson1_react_guarded() -> None:
    s = summary("react")
    assert round(s.success * 24) == 24
    assert (round(s.exact, 3), round(s.correct, 3)) == (0.958, 1.0)
    assert (round(s.avg_steps, 2), round(s.avg_model_calls, 2), round(s.avg_tokens)) == (3.25, 4.25, 3244)


def test_lesson1_duplicate_agreement() -> None:
    assert task_run("react", "task04", OFF).short == "2 million"
    assert task_run("react", "task22").short == "unknown"


# --------------------------------------------------------------------------- lesson 8.2


def test_lesson2_numbers() -> None:
    assert sum(g.result.usage.calls for g in graded("react")) == 102
    r = task_run("rewoo", "task06")
    assert (r.short, r.usage.calls, r.tool_calls, r.usage.tokens) == ("30", 2, 6, 1654)
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (1476, 178)
    p = task_run("plan", "task23")
    assert (p.short, p.usage.calls, p.tool_calls, p.usage.tokens) == ("unknown", 4, 7, 3246)
    assert (p.usage.prompt_tokens, p.usage.completion_tokens) == (3070, 176)


def test_lesson2_defaults_and_replans() -> None:
    import inspect

    assert inspect.signature(run_reflexion).parameters["max_trials"].default == 3
    search = TreeSearch(WebTools(ON))
    assert (search.branching, search.beam) == (3, 2)
    replanned = [g.task["id"] for g in graded("plan") if g.result.usage.calls > 2]
    assert replanned == ["task22", "task23"]
    assert task_run("react", "task04", OFF).usage.calls == 4


def test_lesson2_reflexion_trace() -> None:
    r = run_reflexion(TASKS["task04"]["question"], WebTools(OFF))
    assert (r.short, r.usage.calls, r.tool_calls, r.usage.tokens) == ("410,000", 10, 7, 6280)
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (5839, 441)
    assert r.notes[0].startswith("trial 1: '2 million'") and r.notes[2] == "trial 2: '410,000' | problems: none"


def test_lesson2_comparison_table() -> None:
    assert row(summary("react")) == (1.0, 4.25, 3244, 3.25)
    assert row(summary("rewoo")) == (1.0, 2.0, 1246, 3.88)
    assert row(summary("plan")) == (1.0, 2.17, 1441, 4.21)
    assert row(summary("reflexion")) == (1.0, 5.0, 4059, 3.83)
    assert row(summary("react", OFF))[:3] == (0.917, 4.79, 3536)
    assert row(summary("rewoo", OFF))[:3] == (0.917, 2.0, 1134)
    assert row(summary("plan", OFF))[:3] == (0.917, 2.17, 1303)
    assert row(summary("reflexion", OFF))[:3] == (1.0, 6.0, 4516)
    assert round(summary("rewoo").avg_tokens / summary("react").avg_tokens, 2) == 0.38
    for agent in ("react", "rewoo", "plan"):
        assert summary(agent, OFF).failed_ids == ("task04", "task19")


# --------------------------------------------------------------------------- lesson 8.3


def test_lesson3_task04_scores() -> None:
    r = run_tree(TASKS["task04"]["question"], WebTools(OFF))
    assert r.short == "410,000"
    assert [n.rsplit("score ", 1)[1] for n in r.notes if n.startswith("depth")] == ["0.050", "0.050", "0.500"]
    assert (r.usage.calls, r.tool_calls, r.usage.tokens) == (3, 4, 657)
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (642, 15)


def test_lesson3_task02_scores() -> None:
    naive = run_tree(TASKS["task02"]["question"], WebTools(OFF))
    assert naive.short == "200"
    assert [n.rsplit("score ", 1)[1] for n in naive.notes if n.startswith("depth")] == ["0.500", "0.750", "0.750"]
    guarded = run_tree(TASKS["task02"]["question"], WebTools(ON))
    assert guarded.short == "250" and any(n.endswith("value=1.000") for n in guarded.notes)
    wide = run_tree(TASKS["task03"]["question"], WebTools(OFF), branching=5)
    assert wide.short == "2023-10-02"


def test_lesson3_costs() -> None:
    s = summary("tree")
    assert (round(s.avg_tokens), round(s.avg_steps, 2), round(s.avg_model_calls, 2)) == (1342, 4.79, 4.75)
    r = run_tree(TASKS["task06"]["question"], WebTools(ON), strategy="mcts")
    assert (r.usage.calls, r.tool_calls, r.usage.tokens) == (12, 8, 3903)
    assert TreeSearch(WebTools(ON)).iterations == 12


@pytest.mark.parametrize(
    ("branching", "ok", "tools", "calls", "tokens", "failed"),
    [
        (1, 22, 2.50, 1.29, 350, ("task04", "task19")),
        (2, 22, 3.71, 3.17, 922, ("task04", "task19")),
        (3, 22, 4.92, 4.75, 1369, ("task02", "task19")),
        (5, 21, 7.29, 7.92, 2226, ("task02", "task03", "task19")),
    ],
)
def test_lesson3_branching_sweep(branching: int, ok: int, tools: float, calls: float, tokens: int, failed: tuple) -> None:
    s = summary("tree", OFF, branching=branching)
    assert round(s.success * 24) == ok
    assert (round(s.avg_steps, 2), round(s.avg_model_calls, 2), round(s.avg_tokens)) == (tools, calls, tokens)
    assert s.failed_ids == failed


def test_lesson3_tree_ablation() -> None:
    assert summary("tree", Guards(dedup=False)).failed_ids == ()
    assert summary("tree", Guards(credibility=False)).failed_ids == ()
    assert summary("tree", Guards(dedup=False, credibility=False)).failed_ids == ("task02",)
    assert summary("tree", Guards(sanitize=False)).failed_ids == ("task19",)


# --------------------------------------------------------------------------- lesson 8.4


def test_lesson4_overhead_table() -> None:
    react, multi = summary("react"), summary("multi")
    no_critic = summary("multi", critic=False)
    assert (row(no_critic), no_critic.redundant_calls) == ((1.0, 6.79, 3498, 3.5), 3)
    assert (row(multi), multi.redundant_calls) == ((1.0, 8.0, 4211, 7.12), 54)
    assert react.redundant_calls == 0


def test_lesson4_task06_and_task05() -> None:
    system = MultiAgentSystem(WebTools(ON))
    r = system.run(TASKS["task06"]["question"])
    assert (r.short, r.usage.calls, r.tool_calls, r.usage.tokens, len(system.directory.wire)) == (
        "30", 12, 12, 6954, 8)
    system = MultiAgentSystem(WebTools(ON))
    r = system.run(TASKS["task05"]["question"])
    assert (r.short, r.usage.calls, r.tool_calls, r.usage.tokens) == ("Chile", 6, 5, 2755)
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, len(system.directory.wire)) == (2485, 270, 4)


def test_lesson4_dirty_web_table() -> None:
    assert summary("multi", OFF, critic=False).failed_ids == ("task04",)
    assert summary("multi", OFF).failed_ids == ("task02",)
    assert summary("multi", OFF, structured_handoff=False).failed_ids == ("task02", "task19")
    r = MultiAgentSystem(WebTools(OFF), structured_handoff=False).run(TASKS["task19"]["question"])
    assert "gadgetdeals" in r.answer_text and "Corvid Thermo X" in r.answer_text
    checked = MultiAgentSystem(WebTools(OFF)).run(TASKS["task02"]["question"])
    assert checked.short == "200" and "devices:HomeMesh 2.0: overruled" in checked.notes


# --------------------------------------------------------------------------- lesson 8.5


def test_lesson5_react_report() -> None:
    s = summary("react")
    assert (s.unknown_correct, s.false_unknown) == ((2, 2), 0)
    assert (round(s.citation_precision, 3), s.citations) == (0.926, 27)
    assert round(s.citation_precision * s.citations) == 25
    bad = [g.task["id"] for g in graded("react") for ok in g.citations if not ok]
    assert bad == ["task06", "task06"]
    assert (s.tool_errors, s.redundant_calls, s.violations, s.foreign_domains) == (0, 0, 0, 0)
    assert s.by_type == {
        "aggregation": (1, 1), "comparison": (1, 1), "contradiction": (1, 1), "cross_lingual": (1, 1),
        "injection": (1, 1), "multi_hop": (6, 6), "recency": (1, 1), "single_hop": (10, 10),
        "unanswerable": (2, 2),
    }
    naive = summary("react", OFF)
    assert (round(naive.exact, 3), round(naive.correct, 3), round(naive.citation_precision, 3)) == (
        0.917, 0.958, 0.889)
    assert (naive.tool_errors, naive.redundant_calls, naive.violations, naive.foreign_domains) == (0, 2, 2, 1)


def test_lesson5_ablation_table() -> None:
    expected = {
        ON: (1.0, 0, 3.25, ()),
        Guards(dedup=False): (1.0, 0, 3.25, ()),
        Guards(credibility=False): (0.875, 2, 3.79, ("task02", "task04", "task21")),
        Guards(recency=False): (1.0, 0, 3.25, ()),
        Guards(sanitize=False): (0.958, 1, 3.25, ("task19",)),
        OFF: (0.917, 2, 3.79, ("task04", "task19")),
    }
    for guards, want in expected.items():
        s = summary("react", guards)
        assert (round(s.success, 3), s.violations, round(s.avg_steps, 2), s.failed_ids) == want


def test_lesson5_credibility_trap_and_recency_probe() -> None:
    assert task_run("react", "task21", Guards(credibility=False)).short == "23.5"
    assert task_run("react", "task04", Guards(credibility=False)).short == "2 million"
    q = "How much standby power does the Corvid Thermo X draw?"
    assert run_agent("react", q, ON).short == "0.4 W"
    stale = run_agent("react", q, Guards(recency=False))
    assert (stale.short, stale.usage.calls, stale.tool_calls, stale.usage.tokens) == ("0.9 W", 4, 3, 2728)
    assert (stale.usage.prompt_tokens, stale.usage.completion_tokens) == (2529, 199)


def test_lesson5_judge_agreement() -> None:
    rows = run_probes()
    assert len(rows) == 12
    assert [agreement(rows, g) for g in ("exact", "lenient", "judge")] == [9, 8, 9]
    assert [agreement(rows, g, True) for g in ("exact", "lenient", "judge")] == [9, 9, 10]


def test_lesson5_replay(tmp_path: Path) -> None:
    path = tmp_path / "react-naive.jsonl"
    save_run(path, graded("react", OFF), OFF)
    again = summarize(replay(path))
    assert round(again.success * 24) == 22 and again.failed_ids == ("task04", "task19")


# --------------------------------------------------------------------------- module README, quiz, lab README


def test_compare_tables() -> None:
    assert row(summary("mcts"))[1:3] == (4.62, 1258)
    assert row(summary("tree", OFF))[:3] == (0.917, 4.75, 1369)
    assert row(summary("mcts", OFF))[:3] == (0.917, 4.83, 1330)
    assert row(summary("multi", OFF))[:3] == (0.958, 8.54, 4427)
    assert row(summary("react", max_steps=4))[0] == 0.792
    assert summary("react", max_steps=4).failed_ids == ("task06", "task08", "task11", "task18", "task24")


# --------------------------------------------------------------------------- lab README "Break it"


def test_break_it_sanitiser_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    from m08_web_agent import sanitize as sanitize_module

    patterns = list(sanitize_module.INJECTION_PATTERNS)
    monkeypatch.setattr(sanitize_module, "INJECTION_PATTERNS", patterns[1:])
    assert summarize(run_eval("react", ON)).failed_ids == ()
    monkeypatch.setattr(sanitize_module, "INJECTION_PATTERNS", patterns[2:])
    s = summarize(run_eval("react", ON))
    assert (s.failed_ids, s.violations) == (("task19",), 1)


def test_break_it_pesos_accept_euros(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from m08_web_agent import skills

    old = skills.ATTRIBUTES["price_mxn"]
    patched = dict(skills.ATTRIBUTES)
    patched["price_mxn"] = replace(old, context=("€",), pattern=r"€\s?(\d+(?:\.\d+)?)", template="€{}")
    monkeypatch.setattr(skills, "ATTRIBUTES", patched)
    s = summarize(run_eval("react", ON))
    assert (s.failed_ids, s.unknown_correct) == (("task22",), (1, 2))


def test_break_it_value_model_without_spam_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from m08_web_agent import tree

    class NeverSpam:
        is_spam = False

    monkeypatch.setattr(tree, "text_spam_signals", lambda *_a: NeverSpam())
    s = summarize(run_eval("tree", OFF))
    assert s.failed_ids == ("task02", "task04", "task19")
