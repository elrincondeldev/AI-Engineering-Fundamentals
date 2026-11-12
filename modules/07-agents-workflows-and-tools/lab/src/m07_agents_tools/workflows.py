"""Five workflow patterns on Hearthwise tasks: code decides the path, the LLM fills in steps.

- prompt chaining with gates: :func:`chain`
- routing: :func:`route_tickets`
- parallelization by sectioning: :func:`section_compare`; by voting: :func:`vote_tickets`
- reflection (evaluator-optimizer): :func:`reflect`
- orchestrator-worker: :func:`orchestrate`

Every function returns a small report with the number of LLM calls, the number
of *sequential rounds* (the critical path that drives latency) and tokens.
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import kb
from .data import iter_csv
from .providers import FakeProvider, LLMProvider, Message
from .store import DEMO_CUSTOMER, NotFoundError, OrderStore
from .workflow_model import budget_of, strip_quoted, workflow_model

ORDER_PATTERN = r"^HW-\d{6}$"
CUSTOMER_MESSAGES = [
    "Hi, can I still return the Ember T3 from order HW-310042? It has been about three weeks.",
    "The Glow bulbs from order HW-310077 are too warm for my taste, I want to send them back.",
    "Where is my camera? Order HW-310108.",
    "I want to return my plug but I do not have the order number at hand.",
    "Return order 310155 please.",
    "Can you check order HW-418230 for me?",
]
LANE_OF_INTENT = {
    "order_status": "orders", "return_request": "orders",
    "connectivity_issue": "devices", "product_question": "devices", "warranty_claim": "devices",
    "billing_question": "account", "account_access": "account", "cancel_subscription": "account",
}
MODEL_TIER = {"orders": "small", "account": "small", "devices": "large"}


@dataclass
class Meter:
    """Wraps a provider: one call = one prompt template + one input. Counts calls and tokens."""

    provider: LLMProvider = field(default_factory=lambda: FakeProvider(responder=workflow_model))
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def ask(self, task: str, content: str) -> str:
        reply = self.provider.complete([Message("system", f"TASK: {task}"), Message("user", content)])
        with self._lock:  # workers may call in parallel
            self.llm_calls += 1
            self.prompt_tokens += reply.prompt_tokens
            self.completion_tokens += reply.completion_tokens
        return reply.text

    def usage(self) -> dict[str, int]:
        return {"llm_calls": self.llm_calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


# --------------------------------------------------------------------------- #
# 1. Prompt chaining with gates                                               #
# --------------------------------------------------------------------------- #


def chain_one(meter: Meter, store: OrderStore, message: str) -> dict[str, Any]:
    """extract (LLM) -> gate -> lookup (code) -> draft (LLM) -> gate."""
    fields = json.loads(meter.ask("extract", message))
    order_id = fields.get("order_id")
    if not order_id or not re.match(ORDER_PATTERN, order_id):  # gate 1: stop early, ask the customer
        return {"outcome": "asked_for_order_id", "reply": "Could you share your order number (HW-xxxxxx)?"}
    try:
        order = store.lookup(order_id, DEMO_CUSTOMER)
    except NotFoundError:
        return {"outcome": "not_found", "reply": f"I could not find {order_id} in your account."}
    reply = meter.ask("draft_reply", json.dumps({"intent": fields["intent"], "order": order}))
    if order_id not in reply or "14 days" in reply:  # gate 2: check the draft before sending
        return {"outcome": "needs_human", "reply": reply}
    return {"outcome": "replied", "reply": reply}


def chain(messages: list[str] | None = None) -> dict[str, Any]:
    meter, store = Meter(), OrderStore()
    results = [chain_one(meter, store, m) for m in messages or CUSTOMER_MESSAGES]
    outcomes = Counter(r["outcome"] for r in results)
    return {"results": results, "outcomes": dict(outcomes), "sequential_rounds": 2, **meter.usage()}


# --------------------------------------------------------------------------- #
# 2. Routing                                                                  #
# --------------------------------------------------------------------------- #


def load_tickets() -> list[dict[str, str]]:
    return list(iter_csv("tickets/tickets.csv"))


def ticket_text(ticket: dict[str, str], strip: bool = True) -> str:
    text = f"{ticket['subject']}\n{ticket['body']}"
    return strip_quoted(text) if strip else text  # a cheap code step before the LLM step


def route_tickets(tickets: list[dict[str, str]] | None = None, strip: bool = True) -> dict[str, Any]:
    """Classify each ticket into a lane, then send it to that lane's handler and model tier."""
    meter = Meter()
    tickets = tickets if tickets is not None else load_tickets()
    lanes: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    correct = 0
    for t in tickets:
        lane = meter.ask("route", ticket_text(t, strip))
        lanes[lane] += 1
        correct += lane == LANE_OF_INTENT[t["intent"]]
        if lane != "human":  # a human queue costs no LLM call
            tiers[MODEL_TIER[lane]] += 1
            meter.ask(f"answer_{lane}", t["body"])
    routed = len(tickets) - lanes["human"]
    return {"tickets": len(tickets), "lanes": dict(lanes), "accuracy": round(correct / len(tickets), 3),
            "accuracy_when_routed": round(correct / routed, 3), "tiers": dict(tiers),
            "sequential_rounds": 2, **meter.usage()}


# --------------------------------------------------------------------------- #
# 3. Parallelization: sectioning and voting                                   #
# --------------------------------------------------------------------------- #


def section_compare(slugs: list[str] | None = None) -> dict[str, Any]:
    """One worker per product, all at once, then one aggregation call."""
    meter = Meter()
    slugs = slugs or kb.PRODUCT_SLUGS

    def worker(slug: str) -> str:
        return meter.ask("product_row", json.dumps({**kb.get_product(slug), "slug": slug}))

    with ThreadPoolExecutor(max_workers=len(slugs)) as pool:
        rows = list(pool.map(worker, slugs))  # round 1: all sections in parallel
    table = meter.ask("aggregate_table", "\n".join(rows))  # round 2
    return {"table": table, "sections": len(slugs), "sequential_rounds": 2,
            "sequential_rounds_if_serial": len(slugs) + 1, **meter.usage()}


def vote_tickets(k: int, noise: float = 0.3, tickets: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Ask the noisy router ``k`` times per ticket and keep the majority answer."""
    meter = Meter()
    tickets = tickets if tickets is not None else load_tickets()
    correct = 0
    for t in tickets:
        text = ticket_text(t)
        votes = [meter.ask(f"route_noisy\nSAMPLE {s}\nNOISE {noise}", text) for s in range(k)]
        winner = Counter(votes).most_common(1)[0][0]
        correct += winner == LANE_OF_INTENT[t["intent"]]
    return {"k": k, "noise": noise, "accuracy": round(correct / len(tickets), 3), **meter.usage()}


# --------------------------------------------------------------------------- #
# 4. Reflection (evaluator-optimizer)                                         #
# --------------------------------------------------------------------------- #

REFLECT_QUESTION = "I opened my Ember T3 three weeks ago (order HW-310042). Can I still return it?"


def reflect(question: str = REFLECT_QUESTION, max_rounds: int = 3) -> dict[str, Any]:
    """Draft, evaluate against evidence and a rubric, revise with the feedback, stop when it passes."""
    meter = Meter()
    order = OrderStore().lookup("HW-310042", DEMO_CUSTOMER)
    evidence = {"order_id": order["order_id"], **order["return"]}  # type: ignore[dict-item]
    feedback: list[str] = []
    history = []
    for round_no in range(1, max_rounds + 1):
        draft = meter.ask("draft_policy_answer",
                          json.dumps({"question": question, "feedback": feedback, "evidence": evidence}))
        verdict = json.loads(meter.ask("evaluate", json.dumps({"draft": draft, "evidence": evidence})))
        history.append({"round": round_no, "draft": draft, "issues": verdict["issues"]})
        if verdict["pass"]:
            break
        feedback = verdict["issues"]
    return {"rounds": len(history), "passed": not history[-1]["issues"], "history": history,
            "sequential_rounds": 2 * len(history), **meter.usage()}


# --------------------------------------------------------------------------- #
# 5. Orchestrator-worker                                                      #
# --------------------------------------------------------------------------- #

ORCHESTRATE_REQUESTS = [
    "A customer renting a flat wants smart lights and heating control. Budget €200.",
    "A customer wants motion-activated lights in the hallway. Budget €100.",
]


def orchestrate(request: str) -> dict[str, Any]:
    """The orchestrator plans subtasks at run time, may add more after reading results, then synthesises."""
    meter = Meter()

    def worker(slug: str) -> dict[str, Any]:
        return json.loads(meter.ask("product_worker", json.dumps({**kb.get_product(slug), "slug": slug})))

    plan: list[str] = json.loads(meter.ask("plan", request))
    results: list[dict[str, Any]] = []
    rounds, todo = 1, plan
    while todo:
        with ThreadPoolExecutor(max_workers=len(todo)) as pool:
            results.extend(pool.map(worker, todo))
        todo = json.loads(meter.ask("replan", json.dumps({"request": request, "results": results})))
        rounds += 2
    answer = meter.ask("synthesise", json.dumps({"budget_eur": budget_of(request), "results": results}))
    return {"request": request, "plan": plan, "workers": [r["product"] for r in results], "answer": answer,
            "sequential_rounds": rounds + 1, **meter.usage()}
