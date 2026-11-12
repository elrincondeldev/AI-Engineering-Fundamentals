"""Behaviour of the five workflow patterns and the agency ladder."""

from __future__ import annotations

from m07_agents_tools import workflows as wf
from m07_agents_tools.agency import ladder
from m07_agents_tools.workflow_model import noisy_route, route, strip_quoted


def test_chain_gate_stops_early_without_a_valid_order_id() -> None:
    report = wf.chain()
    outcomes = [r["outcome"] for r in report["results"]]
    assert outcomes == ["replied", "replied", "replied", "asked_for_order_id", "asked_for_order_id", "not_found"]
    # a stopped message costs one LLM call (extract) instead of two
    assert report["llm_calls"] == 6 + outcomes.count("replied")


def test_chain_second_gate_catches_a_bad_draft() -> None:
    class StaleDrafts(wf.Meter):
        def ask(self, task: str, content: str) -> str:
            text = super().ask(task, content)
            return "You have 14 days to return it." if task == "draft_reply" else text

    result = wf.chain_one(StaleDrafts(), wf.OrderStore(), wf.CUSTOMER_MESSAGES[0])
    assert result["outcome"] == "needs_human"


def test_router_basics() -> None:
    assert route("Where is my order? The tracking page has not changed.") == "orders"
    assert route("I can't log in, my password is wrong") == "account"
    assert route("The bulb keeps disconnecting from Wi-Fi") == "devices"
    assert route("Hello there") == "human"
    assert strip_quoted("Hi\n> Could you share your order number?\nBye") == "Hi\nBye"


def test_noisy_router_is_deterministic() -> None:
    text = "Where is my order?"
    assert [noisy_route(text, s, 0.3) for s in range(10)] == [noisy_route(text, s, 0.3) for s in range(10)]
    assert all(noisy_route(text, s, 0.0) == "orders" for s in range(10))


def test_voting_cannot_beat_the_noise_free_router() -> None:
    clean = wf.route_tickets()["accuracy"]
    accs = [wf.vote_tickets(k)["accuracy"] for k in (1, 3, 5)]
    assert accs[0] < accs[1] < accs[2] < clean


def test_sectioning_runs_all_sections() -> None:
    report = wf.section_compare()
    assert report["table"].count("\n") == 7  # header, separator and six rows
    assert "| Watcher S1 | €29 | none (HomeMesh radio) | yes |" in report["table"]


def test_reflection_fixes_the_stale_policy() -> None:
    report = wf.reflect()
    first, last = report["history"][0], report["history"][-1]
    assert "14 days" in first["draft"] and "30 days" in last["draft"] and last["issues"] == []


def test_reflection_respects_max_rounds() -> None:
    report = wf.reflect(max_rounds=1)
    assert report["rounds"] == 1 and report["passed"] is False


def test_orchestrator_adds_the_hub_after_reading_worker_results() -> None:
    r1, r2 = (wf.orchestrate(q) for q in wf.ORCHESTRATE_REQUESTS)
    assert r1["workers"] == r1["plan"]
    assert r2["workers"] == r2["plan"] + ["hearth-hub"]


def test_ladder_increases_autonomy_and_cost() -> None:
    rungs = ladder()
    assert [r.llm_calls for r in rungs] == sorted(r.llm_calls for r in rungs)
    assert [r.tokens for r in rungs] == sorted(r.tokens for r in rungs)
    assert [r.data_changed for r in rungs] == [False, False, False, True]
