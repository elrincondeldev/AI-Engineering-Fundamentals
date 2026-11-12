"""Control loops: ReAct, ReWOO, plan-and-execute, Reflexion, tree search and multi-agent."""

from __future__ import annotations

from m08_web_agent.agents import parse_action, run_react, run_rewoo
from m08_web_agent.model import RULE_CREDIBLE, ScriptedModel, read_context
from m08_web_agent.multi import STATES, MultiAgentSystem, jsonrpc_envelope
from m08_web_agent.providers import FakeProvider, Message
from m08_web_agent.reflexion import run_reflexion, self_evaluate
from m08_web_agent.search import Guards
from m08_web_agent.tools import WebTools
from m08_web_agent.tree import run_tree

Q_PRICE = "How much more expensive is the Corvid Thermo X than the Hearthwise Ember T3, in euros?"
Q_SPAM = "How many smart thermostats were sold in Spain in 2025?"


def test_parse_action() -> None:
    assert parse_action("Thought: x\nAction: search_web[ember t3]") == ("search_web", "ember t3")
    assert parse_action("Thought: done\nFinal Answer: 30\nSources: none")[0] == "final"
    assert parse_action("I think the answer is 30")[0] == "invalid"


def test_react_trajectory_and_answer() -> None:
    r = run_react(Q_PRICE, WebTools(Guards()))
    assert r.short == "30"
    assert [s.tool for s in r.steps][:2] == ["search_web", "open_page"]
    assert r.usage.calls == len(r.steps) + 1  # one model call per step plus the final answer
    assert r.prompt_sizes == sorted(r.prompt_sizes)  # the scratchpad only grows


def test_react_respects_the_step_budget() -> None:
    q = "Which bulb gives more lumens per euro: the Glow A19 or the Brightline Bulb Pro?"
    r = run_react(q, WebTools(Guards()), max_steps=3)
    assert r.stopped == "budget" and r.tool_calls == 3 and r.short == "unknown"


def test_model_reads_rules_from_the_transcript() -> None:
    ctx = read_context([Message("system", f"- {RULE_CREDIBLE}"), Message("user", "Question: q")])
    assert ctx.prefer_credible and not ctx.prefer_recent


def test_invalid_model_output_is_an_observed_error() -> None:
    replies = iter(["no idea", "Thought: ok\nFinal Answer: unknown\nExplanation: x\nSources: none"])
    provider = FakeProvider(responder=lambda _m: next(replies))
    r = run_react("anything?", WebTools(Guards()), provider)
    assert r.tool_errors == 1 and r.short == "unknown"


def test_rewoo_uses_two_model_calls() -> None:
    r = run_rewoo(Q_PRICE, WebTools(Guards()))
    assert r.short == "30" and r.usage.calls == 2
    assert r.notes[0].startswith("Plan: Find the price of the Corvid Thermo X.")


def test_plan_and_execute_replans_only_when_facts_are_missing() -> None:
    found = run_rewoo(Q_PRICE, WebTools(Guards()), replan=True)
    missing = run_rewoo("How many Lookout C2 cameras has Hearthwise sold?", WebTools(Guards()), replan=True)
    assert found.usage.calls == 2 and missing.usage.calls == 4 and missing.short == "unknown"


def test_reflexion_recovers_from_spam_without_guards() -> None:
    first = run_react(Q_SPAM, WebTools(Guards.none()))
    assert first.short == "2 million"
    assert any(p.startswith("[spam]") for p in self_evaluate(first))
    r = run_reflexion(Q_SPAM, WebTools(Guards.none()))
    assert r.short == "410,000"
    assert any(n.startswith("reflection: I relied on a page that looks like spam") for n in r.notes)


def test_tree_search_compares_alternatives() -> None:
    greedy = run_tree(Q_SPAM, WebTools(Guards.none()), branching=1)
    wide = run_tree(Q_SPAM, WebTools(Guards.none()), branching=3)
    assert greedy.short == "2 million" and wide.short == "410,000"
    assert wide.tool_calls > greedy.tool_calls


def test_mcts_is_deterministic() -> None:
    a = run_tree(Q_PRICE, WebTools(Guards()), strategy="mcts", seed=7)
    b = run_tree(Q_PRICE, WebTools(Guards()), strategy="mcts", seed=7)
    assert a.answer_text == b.answer_text and a.usage.calls == b.usage.calls


def test_multi_agent_tasks_follow_a2a_style_lifecycle() -> None:
    system = MultiAgentSystem(WebTools(Guards()))
    r = system.run(Q_PRICE)
    assert r.short == "30"
    assert set(system.directory.cards) == {"researcher", "fact_checker"}
    requests = [e for e in system.directory.wire if "method" in e]
    responses = [e for e in system.directory.wire if "result" in e]
    assert len(requests) == len(responses) == 4
    assert all(e["jsonrpc"] == "2.0" for e in system.directory.wire)
    assert all(e["result"]["state"] == "completed" for e in responses)
    assert "input-required" in STATES
    assert jsonrpc_envelope("m", {}, 1) == {"jsonrpc": "2.0", "id": 1, "method": "m", "params": {}}


def test_free_text_handoff_lets_an_injection_through() -> None:
    q = "What do forum users recommend to fix an Ember T3 that keeps dropping Wi-Fi?"
    structured = MultiAgentSystem(WebTools(Guards.none())).run(q)
    free_text = MultiAgentSystem(WebTools(Guards.none()), structured_handoff=False).run(q)
    assert "gadgetdeals" not in structured.answer_text
    assert "gadgetdeals" in free_text.answer_text


def test_scripted_model_is_stateless() -> None:
    model = ScriptedModel()
    msgs = [Message("system", "rules"), Message("user", f"Question: {Q_PRICE}")]
    assert model.react_step(msgs) == model.react_step(msgs)
