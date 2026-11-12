"""The agency ladder: one customer request solved at four levels of autonomy.

Same request, same data, same fake model family. What changes is who decides
the next step (your code or the model) and what the model is allowed to touch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .executor import Permission, ToolExecutor
from .loop import run_tool_loop
from .providers import FakeProvider, Message
from .scripted import make_completion, scripted_provider
from .store import DEMO_CUSTOMER, OrderStore
from .tools import Session, build_toolbox

LADDER_REQUEST = "I bought an Ember T3 (order HW-310042) and changed my mind. Please start a return."


@dataclass(frozen=True)
class Rung:
    level: str
    who_decides: str
    llm_calls: int
    tool_calls: int
    tokens: int
    data_changed: bool
    answer: str


def single_call(request: str = LADDER_REQUEST) -> Rung:
    """Level 1: one LLM call, no tools, no data. It can only give generic advice."""
    generic = ("Hearthwise accepts returns within 30 days of delivery. I cannot see your order, so please "
               "check the delivery date in the app under Account > Orders.")
    provider = FakeProvider(responder=lambda msgs: make_completion(msgs, text=generic))
    reply = provider.complete([Message("system", "You are Hearthwise Assist."), Message("user", request)])
    return Rung("1 single LLM call", "nobody", 1, 0, reply.prompt_tokens + reply.completion_tokens, False,
                reply.text)


def workflow(request: str = LADDER_REQUEST) -> Rung:
    """Level 2: a fixed path written in code. Code extracts and looks up; the LLM only writes the reply."""
    store = OrderStore()
    order_id = re.search(r"HW-\d{6}", request).group(0)  # code decides: always look the order up
    order = store.lookup(order_id, DEMO_CUSTOMER)
    check: dict[str, Any] = order["return"]  # type: ignore[assignment]
    text = (f"Order {order_id} can be returned ({check['days_left']} days left). Open the app, go to "
            "Account > Orders and tap Return an item.")
    provider = FakeProvider(responder=lambda msgs: make_completion(msgs, text=text))
    reply = provider.complete([Message("system", "Write a short reply from these facts."),
                               Message("user", json.dumps(order))])
    return Rung("2 workflow", "your code", 1, 1, reply.prompt_tokens + reply.completion_tokens, False,
                reply.text)


def _loop_rung(level: str, allow_writes: bool, request: str) -> Rung:
    policy = {} if allow_writes else {"create_return": Permission.DENY}
    executor = ToolExecutor(build_toolbox(OrderStore()), Session(DEMO_CUSTOMER), policy=policy,
                            approver=lambda call: True)  # stands in for a person clicking "Approve"
    trace = run_tool_loop(scripted_provider(executor.specs()), executor, request, max_steps=6)
    changed = any(r.name == "create_return" and r.ok for r in trace.results)
    return Rung(level, "the model", trace.llm_calls, len(trace.results),
                trace.prompt_tokens + trace.completion_tokens, changed, trace.final_text)


def ladder(request: str = LADDER_REQUEST) -> list[Rung]:
    """Run the same request at levels 1 to 4 (level 5, multi-agent, is module 08)."""
    return [
        single_call(request),
        workflow(request),
        _loop_rung("3 tool-using (read-only)", allow_writes=False, request=request),
        _loop_rung("4 agent (can act, approved)", allow_writes=True, request=request),
    ]
