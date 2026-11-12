"""The executor (permissions, validation, approval, retries, timeouts, idempotency) and the tool loop."""

from __future__ import annotations

import json
import threading
import time

from m07_agents_tools.executor import Permission, ToolExecutor, idempotency_key
from m07_agents_tools.loop import run_tool_loop
from m07_agents_tools.providers import ToolCall
from m07_agents_tools.scripted import scripted_provider
from m07_agents_tools.store import DEMO_CUSTOMER, OrderStore
from m07_agents_tools.tools import Session, Tool, build_toolbox, inject_faults

SESSION = Session(DEMO_CUSTOMER)
RETURN_ARGS = {"order_id": "HW-310042", "sku": "HW-EMB-T3", "quantity": 1, "reason": "changed_mind"}


def make(store: OrderStore | None = None, **kwargs) -> ToolExecutor:
    kwargs.setdefault("sleep", lambda s: None)
    return ToolExecutor(build_toolbox(store or OrderStore()), SESSION, **kwargs)


def test_unknown_tool_and_invalid_arguments_go_back_as_results() -> None:
    ex = make()
    unknown = ex.execute(ToolCall("c1", "delete_everything", {}))
    assert unknown.status == "unknown_tool" and "lookup_order" in unknown.error
    bad = ex.execute(ToolCall("c2", "lookup_order", {"order_id": "310042"}))
    assert bad.status == "invalid_arguments" and bad.attempts == 0
    message = bad.to_message()
    assert message.role == "tool" and message.tool_call_id == "c2"
    assert json.loads(message.content)["error"]["type"] == "invalid_arguments"


def test_side_effects_need_approval_by_default() -> None:
    store = OrderStore()
    no_one_asked = make(store).execute(ToolCall("c1", "create_return", RETURN_ARGS))
    assert no_one_asked.status == "rejected" and store.returns == []
    approved = make(store, approver=lambda call: True).execute(ToolCall("c2", "create_return", RETURN_ARGS))
    assert approved.ok and len(store.returns) == 1


def test_denied_tools_are_hidden_and_refused() -> None:
    ex = make(policy={"create_return": Permission.DENY}, approver=lambda call: True)
    assert "create_return" not in [s["function"]["name"] for s in ex.specs()]
    assert ex.execute(ToolCall("c1", "create_return", RETURN_ARGS)).status == "denied"


def test_transient_errors_are_retried_with_backoff() -> None:
    waits: list[float] = []
    tools = inject_faults(build_toolbox(OrderStore()), {"lookup_order": ["transient", "transient"]})
    ex = ToolExecutor(tools, SESSION, backoff_s=0.1, sleep=waits.append)
    result = ex.execute(ToolCall("c1", "lookup_order", {"order_id": "HW-310108"}))
    assert result.ok and result.attempts == 3
    assert waits == [0.1, 0.2]  # exponential backoff


def test_retries_give_up_after_the_budget() -> None:
    tools = inject_faults(build_toolbox(OrderStore()), {"lookup_order": ["transient"] * 3})
    result = ToolExecutor(tools, SESSION, sleep=lambda s: None).execute(
        ToolCall("c1", "lookup_order", {"order_id": "HW-310108"}))
    assert result.status == "error" and result.attempts == 3


def test_business_errors_are_not_retried() -> None:
    result = make(approver=lambda c: True).execute(
        ToolCall("c1", "create_return", {**RETURN_ARGS, "order_id": "HW-310077", "sku": "HW-GLW-A19"}))
    assert result.status == "error" and result.attempts == 1 and "49 days" in result.error


def test_timeouts_are_enforced() -> None:
    tools = inject_faults(build_toolbox(OrderStore(), timeout_s=0.05), {"get_product": ["slow"] * 3}, slow_s=0.2)
    result = ToolExecutor(tools, SESSION, max_retries=1, sleep=lambda s: None).execute(
        ToolCall("c1", "get_product", {"product": "ember-t3"}))
    assert result.status == "timeout" and result.attempts == 2


def test_idempotency_key_prevents_double_returns_after_a_timeout() -> None:
    args = {"order_id": "HW-310155", "sku": "HW-SPK-MINI", "quantity": 1, "reason": "changed_mind"}
    for keys, expected in ((True, 1), (False, 2)):
        store = OrderStore()
        tools = inject_faults(build_toolbox(store, timeout_s=0.05), {"create_return": ["slow_after_commit"]},
                              slow_s=0.2)
        ex = ToolExecutor(tools, SESSION, approver=lambda c: True, use_idempotency_keys=keys,
                          sleep=lambda s: None)
        result = ex.execute(ToolCall("c1", "create_return", args))
        assert result.ok and result.attempts == 2
        assert len(store.returns) == expected


def test_idempotency_key_is_stable() -> None:
    call = ToolCall("x", "create_return", RETURN_ARGS)
    assert idempotency_key(SESSION, call) == idempotency_key(SESSION, ToolCall("y", "create_return", RETURN_ARGS))
    assert idempotency_key(SESSION, call) != idempotency_key(Session(DEMO_CUSTOMER, "conv-2"), call)


def test_parallel_calls_run_concurrently_and_keep_order() -> None:
    barrier = threading.Barrier(2, timeout=2)

    def slow(args, session):
        barrier.wait()  # only passes if both calls run at the same time
        return args["n"]

    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    ex = ToolExecutor({"t": Tool("t", "test", schema, slow)}, SESSION)
    start = time.perf_counter()
    results = ex.execute_all([ToolCall("a", "t", {"n": 1}), ToolCall("b", "t", {"n": 2})])
    assert [r.data for r in results] == [1, 2] and [r.call_id for r in results] == ["a", "b"]
    assert time.perf_counter() - start < 2


def test_audit_log_records_every_call() -> None:
    ex = make()
    ex.execute(ToolCall("c1", "lookup_order", {"order_id": "HW-310042"}))
    ex.execute(ToolCall("c2", "create_return", RETURN_ARGS))
    assert [(a["tool"], a["status"]) for a in ex.audit] == [("lookup_order", "ok"), ("create_return", "rejected")]


def run(request: str, **kwargs):
    ex = make(**kwargs)
    return run_tool_loop(scripted_provider(ex.specs()), ex, request)


def test_loop_self_corrects_from_a_validation_error() -> None:
    trace = run("Where is my order 310108?")
    assert [r.status for r in trace.results] == ["invalid_arguments", "ok"]
    assert trace.final_text == "Order HW-310108 is shipped."
    assert trace.llm_calls == 3 and trace.stop_reason == "answered"


def test_loop_issues_parallel_calls_for_a_comparison() -> None:
    trace = run("Compare the Ember T3 and the Lookout C2, please.")
    first_turn = trace.messages[2]
    assert len(first_turn.tool_calls) == 2 and trace.llm_calls == 2
    assert "Ember T3: €149" in trace.final_text and "Lookout C2: €89" in trace.final_text


def test_read_only_session_does_not_act() -> None:
    trace = run("Please return the Ember T3 from order HW-310042.", policy={"create_return": Permission.DENY})
    assert [r.name for r in trace.results] == ["lookup_order"]
    assert "Account > Orders" in trace.final_text


def test_loop_stops_at_max_steps() -> None:
    ex = make()

    def stubborn(messages):
        from m07_agents_tools.scripted import make_completion
        return make_completion(messages, calls=[ToolCall("c", "lookup_order", {"order_id": "HW-310042"})])

    from m07_agents_tools.providers import FakeProvider
    trace = run_tool_loop(FakeProvider(responder=stubborn), ex, "loop forever", max_steps=3)
    assert trace.stop_reason == "max_steps" and trace.llm_calls == 3 and len(trace.results) == 3


def test_policy_question_uses_search_and_cites() -> None:
    trace = run("How long do refunds take after a return?")
    assert trace.results[0].name == "search_help_centre"
    assert trace.final_text.endswith("[kb-011]")
