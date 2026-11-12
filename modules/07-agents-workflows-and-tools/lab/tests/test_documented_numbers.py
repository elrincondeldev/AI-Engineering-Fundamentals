"""Every number quoted in the module's Markdown, asserted against a real run of the lab.

If a test here fails after you change the code, update the lessons, README, quiz, cheatsheet and lab
README together with the expected value. Exact equality unless a tolerance is written down.
"""

from __future__ import annotations

import itertools
import json
import re

import pytest

from m07_agents_tools import cli, kb
from m07_agents_tools import workflows as wf
from m07_agents_tools.agency import ladder
from m07_agents_tools.executor import Permission, ToolExecutor
from m07_agents_tools.loop import run_tool_loop
from m07_agents_tools.mcp_client import StdioClient, diff_pins, lint_description, pin_tools
from m07_agents_tools.providers import count_tokens
from m07_agents_tools.scripted import scripted_provider
from m07_agents_tools.store import DEMO_CUSTOMER, TODAY, OrderStore, generate_orders
from m07_agents_tools.tools import Session, build_toolbox

RETURN_REQUEST = "Please return the Ember T3 from order HW-310042, I changed my mind."


# --------------------------------------------------------------------------- #
# Lesson 7.1 and the module README / cheatsheet / quiz Q2                     #
# --------------------------------------------------------------------------- #


def test_ladder_numbers() -> None:
    rungs = ladder()
    assert [r.llm_calls for r in rungs] == [1, 1, 2, 3]
    assert [r.tool_calls for r in rungs] == [0, 1, 1, 2]
    assert [r.tokens for r in rungs] == [63, 154, 1416, 3314]
    assert [r.data_changed for r in rungs] == [False, False, False, True]
    assert rungs[3].tokens / rungs[0].tokens > 50  # "more than 50 times"
    assert "RMA-2A7E64" in rungs[3].answer and "€149" in rungs[3].answer
    assert "(8 days left)" in rungs[1].answer


def test_tool_spec_tokens() -> None:
    tools = build_toolbox()
    per_tool = {name: count_tokens(json.dumps(t.spec())) for name, t in tools.items()}
    assert per_tool == {"get_product": 192, "search_help_centre": 194, "lookup_order": 150, "create_return": 331}
    assert count_tokens(json.dumps([t.spec() for t in tools.values()])) == 872
    assert len(tools) == 4


def test_quiz_q2_share_of_tool_tokens() -> None:
    assert 3 * 872 == 2616
    assert round(2616 / 3314 * 100) == 79


def test_compounding_errors_illustration() -> None:
    assert round(0.95**10, 2) == 0.60


def test_demo_order_facts() -> None:
    store = OrderStore()
    hw42 = store.lookup("HW-310042", DEMO_CUSTOMER)
    assert hw42["delivered_on"] == "2025-10-23" and hw42["return"]["days_since_delivery"] == 22
    assert hw42["return"]["days_left"] == 8
    assert store.lookup("HW-310077", DEMO_CUSTOMER)["return"]["days_since_delivery"] == 49
    assert store.lookup("HW-310155", DEMO_CUSTOMER)["return"]["days_left"] == 19
    assert store.lookup("HW-310108", DEMO_CUSTOMER)["total_eur"] == 89
    assert store.orders["HW-418230"].customer_id == "C-0107"
    assert len(generate_orders()) == 60 and TODAY.isoformat() == "2025-11-14"


# --------------------------------------------------------------------------- #
# Lesson 7.2                                                                  #
# --------------------------------------------------------------------------- #


def test_chain_numbers() -> None:
    report = wf.chain()
    assert len(wf.CUSTOMER_MESSAGES) == 6
    assert report["outcomes"] == {"replied": 3, "asked_for_order_id": 2, "not_found": 1}
    assert report["llm_calls"] == 9 and 2 * len(wf.CUSTOMER_MESSAGES) == 12
    assert report["sequential_rounds"] == 2
    assert report["prompt_tokens"] + report["completion_tokens"] == 729


def test_route_numbers() -> None:
    raw, clean = wf.route_tickets(strip=False), wf.route_tickets()
    assert clean["tickets"] == 1224
    assert raw["accuracy"] == 0.818 and raw["lanes"]["human"] == 165
    assert clean["accuracy"] == 0.848 and clean["lanes"]["human"] == 136
    assert clean["accuracy_when_routed"] == 0.954
    assert clean["lanes"] == {"orders": 344, "devices": 447, "account": 297, "human": 136}
    assert clean["tiers"] == {"small": 641, "large": 447}
    assert clean["llm_calls"] == 2312
    assert clean["prompt_tokens"] + clean["completion_tokens"] == 96096


def test_section_numbers() -> None:
    report = wf.section_compare()
    assert report["sections"] == 6 and report["llm_calls"] == 7
    assert report["sequential_rounds"] == 2 and report["sequential_rounds_if_serial"] == 7
    assert report["prompt_tokens"] + report["completion_tokens"] == 1441
    assert "| Ember T3 | €149 | 2.4 GHz | no |" in report["table"]


def test_vote_numbers() -> None:
    results = {k: wf.vote_tickets(k) for k in (1, 3, 5)}
    assert [results[k]["accuracy"] for k in (1, 3, 5)] == [0.613, 0.697, 0.770]
    assert [results[k]["llm_calls"] for k in (1, 3, 5)] == [1224, 3672, 6120]
    assert results[1]["noise"] == 0.3
    assert results[5]["llm_calls"] / results[1]["llm_calls"] == 5


def test_reflect_numbers() -> None:
    report = wf.reflect()
    assert report["rounds"] == 2 and report["passed"] is True
    assert len(report["history"][0]["issues"]) == 3
    assert report["llm_calls"] == 4 and report["sequential_rounds"] == 4
    assert report["prompt_tokens"] + report["completion_tokens"] == 538
    first, second = report["history"][0]["draft"], report["history"][1]["draft"]
    assert "14 days" in first and "10% restocking fee" in first
    assert "delivered 22 days ago" in second and "8 days left" in second
    assert "longer than 70 words" not in " ".join(report["history"][1]["issues"])
    from m07_agents_tools.workflow_model import _evaluate

    assert "longer than 70 words" in _evaluate("word " * 71 + "30 days [kb-011]")["issues"]
    assert _evaluate(" ".join(["w"] * 68) + " [kb-011]")["issues"] == []


def test_orchestrate_numbers() -> None:
    r1, r2 = (wf.orchestrate(q) for q in wf.ORCHESTRATE_REQUESTS)
    assert r1["answer"].endswith("Total €161, within budget (€200).")
    assert (r1["llm_calls"], r1["sequential_rounds"], len(r1["workers"])) == (5, 4, 2)
    assert r1["prompt_tokens"] + r1["completion_tokens"] == 721
    assert r2["answer"].endswith("Total €110, over budget by €10 (€100).")
    assert (r2["llm_calls"], r2["sequential_rounds"], len(r2["workers"])) == (7, 6, 3)
    assert r2["prompt_tokens"] + r2["completion_tokens"] == 1127


# --------------------------------------------------------------------------- #
# Lesson 7.3 and lab steps 5-6                                                #
# --------------------------------------------------------------------------- #


def demo(capsys: pytest.CaptureFixture[str], name: str, keys: bool = True) -> tuple[dict, str]:
    result = cli.run_demo(name, keys=keys)
    return result, capsys.readouterr().out


def test_fix_args_and_compare(capsys: pytest.CaptureFixture[str]) -> None:
    fix, _ = demo(capsys, "fix-args")
    t = fix["trace"]
    assert (t.llm_calls, len(t.results), t.prompt_tokens, t.completion_tokens) == (3, 2, 3073, 29)
    cmp_, _ = demo(capsys, "compare")
    t = cmp_["trace"]
    assert (t.llm_calls, len(t.results), t.prompt_tokens, t.completion_tokens) == (2, 2, 2275, 65)
    assert len(t.messages[2].tool_calls) == 2  # both calls in one model turn


def test_approval_demos(capsys: pytest.CaptureFixture[str]) -> None:
    ok, _ = demo(capsys, "return")
    assert (ok["trace"].prompt_tokens, ok["trace"].completion_tokens) == (3214, 85)
    assert len(ok["store"].returns) == 1
    denied, out = demo(capsys, "return-denied")
    assert (denied["trace"].prompt_tokens, denied["trace"].completion_tokens) == (3185, 68)
    assert len(denied["store"].returns) == 0 and "Returns in the store: 0" in out
    mine, _ = demo(capsys, "not-yours")
    t = mine["trace"]
    assert (t.llm_calls, len(t.results), t.prompt_tokens, t.completion_tokens) == (2, 1, 1914, 30)


def test_retry_timeout_and_idempotency_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    executor = ToolExecutor(build_toolbox(), Session(DEMO_CUSTOMER))
    assert executor.backoff_s == 0.05 and executor.max_retries == 2
    assert [executor.backoff_s * 2 ** (a - 1) for a in (1, 2)] == [0.05, 0.1]
    assert cli.DEMO_TIMEOUT_S == 0.2 and cli.DEMO_SLOW_S == 0.5
    retries, _ = demo(capsys, "retries")
    assert retries["trace"].results[0].attempts == 3
    timeout, _ = demo(capsys, "timeout")
    assert timeout["trace"].results[0].attempts == 2
    with_keys, _ = demo(capsys, "idempotency")
    without, _ = demo(capsys, "idempotency", keys=False)
    assert len(with_keys["store"].returns) == 1 and len(without["store"].returns) == 2


def test_schema_numbers() -> None:
    params = build_toolbox()["create_return"].parameters
    assert params["properties"]["quantity"]["minimum"] == 1 and params["properties"]["quantity"]["maximum"] == 10
    assert len(params["required"]) == 4
    assert len(build_toolbox()["get_product"].parameters["properties"]["product"]["enum"]) == 6


# --------------------------------------------------------------------------- #
# Lesson 7.4 and lab steps 8-9                                                #
# --------------------------------------------------------------------------- #


def test_mcp_demo_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["mcp-demo"]) == 0
    out = capsys.readouterr().out
    assert "JSON-RPC messages exchanged: 13" in out
    assert "Totals: 3 LLM calls, 2 tool calls, 3214 prompt tokens, 85 completion tokens" in out
    assert "RMA-7DB7F3" in out


def test_lint_and_pin_numbers() -> None:
    approved = pin_tools([t.mcp_spec() for t in build_toolbox().values()])
    with StdioClient(command=[*StdioClient().command, "--poisoned"]) as client:
        client.initialize()
        listed = client.list_tools()
    names = {t["name"] for t in listed}
    findings = {t["name"]: lint_description(t, names) for t in listed}
    assert len(findings["check_stock"]) == 3 and len(findings["create_return"]) == 1
    assert sum(1 for f in findings.values() if f) == 2
    assert diff_pins(approved, pin_tools(listed))["added"] == ["check_stock"]


def test_json_rpc_codes() -> None:
    from m07_agents_tools import mcp_server as s

    assert (s.PARSE_ERROR, s.INVALID_REQUEST, s.METHOD_NOT_FOUND, s.INVALID_PARAMS) == (
        -32700, -32600, -32601, -32602)
    assert StdioClient().timeout_s == 10.0


def test_hidden_elements_and_search_facts() -> None:
    assert sum(d.hidden_removed for d in kb.load_docs()) == 1
    assert [h["id"] for h in kb.search("shipping to Latin America cost")] == ["kb-013"]
    assert [h["id"] for h in kb.search("how long does delivery to Chile take")][:2] == ["kb-013", "kb-031"]


# --------------------------------------------------------------------------- #
# Lab README: solutions and extensions (reference solutions applied in-test)  #
# --------------------------------------------------------------------------- #


class NormalisingMeter(wf.Meter):
    """Same effect as the step 4 solution: six digits become HW-###### before gate 1."""

    def ask(self, task: str, content: str) -> str:
        out = super().ask(task, content)
        if task == "extract":
            fields = json.loads(out)
            if fields["order_id"] and re.fullmatch(r"\d{6}", fields["order_id"]):
                fields["order_id"] = f"HW-{fields['order_id']}"
            out = json.dumps(fields)
        return out


def test_step4_numbers() -> None:
    meter, store = NormalisingMeter(), OrderStore()
    results = [wf.chain_one(meter, store, m) for m in wf.CUSTOMER_MESSAGES]
    outcomes = {o: sum(r["outcome"] == o for r in results) for o in ("replied", "asked_for_order_id", "not_found")}
    assert outcomes == {"replied": 4, "asked_for_order_id": 1, "not_found": 1}
    assert meter.llm_calls == 10 and meter.prompt_tokens + meter.completion_tokens == 921
    assert "19 days left" in results[4]["reply"]


def test_step7_warranty_dates() -> None:
    delivered = OrderStore().orders["HW-310042"].delivered_on
    assert delivered is not None
    assert delivered.replace(year=delivered.year + 2).isoformat() == "2027-10-23"
    assert delivered.replace(year=delivered.year + 3).isoformat() == "2028-10-23"


def test_break4_hidden_text_would_rank_first(monkeypatch: pytest.MonkeyPatch) -> None:
    original = kb._TextExtractor.handle_starttag

    def keep_hidden(self, tag, attrs):  # the Break it 4 change: nothing counts as hidden
        return original(self, tag, [(k, v) for k, v in attrs if k != "style"])

    monkeypatch.setattr(kb._TextExtractor, "handle_starttag", keep_hidden)
    kb.load_docs.cache_clear()
    try:
        assert kb.search("refund verify")[0]["id"] == "kb-025"
    finally:
        monkeypatch.undo()
        kb.load_docs.cache_clear()


def _per_call_prompt_tokens(request: str) -> list[int]:
    executor = ToolExecutor(build_toolbox(OrderStore()), Session(DEMO_CUSTOMER), approver=lambda c: True)
    provider = scripted_provider(executor.specs())
    seen: list[int] = []
    original = provider.complete

    def complete(messages, **kwargs):
        reply = original(messages, **kwargs)
        seen.append(reply.prompt_tokens)
        return reply

    provider.complete = complete  # type: ignore[method-assign]
    run_tool_loop(provider, executor, request)
    return seen


def test_extend4_budget_numbers() -> None:
    compare = _per_call_prompt_tokens(cli.DEMO_REQUESTS["compare"])
    ret = _per_call_prompt_tokens(RETURN_REQUEST)
    assert compare == [932, 1343] and sum(compare) == 2275
    assert ret == [937, 1079, 1198] and list(itertools.accumulate(ret))[1] == 2016
    assert compare[0] <= 2000 < list(itertools.accumulate(ret))[1]


def test_read_only_level_hides_create_return() -> None:
    executor = ToolExecutor(build_toolbox(), Session(DEMO_CUSTOMER), policy={"create_return": Permission.DENY})
    assert [s["function"]["name"] for s in executor.specs()] == ["get_product", "search_help_centre",
                                                                 "lookup_order"]


def test_product_prices_quoted_in_lessons() -> None:
    prices = {kb.get_product(s)["name"]: kb.get_product(s)["price_eur"] for s in kb.PRODUCT_SLUGS}
    assert prices == {"Ember T3": 149, "Glow A19": 12, "Spark Mini": 19, "Watcher S1": 29,
                      "Lookout C2": 89, "Hearth Hub": 69}
