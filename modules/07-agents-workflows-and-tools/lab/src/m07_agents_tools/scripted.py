"""A rule-based stand-in for a tool-calling LLM, used as a ``FakeProvider`` responder.

It reads the conversation the same way a real model would (the user's request
plus the ``tool`` messages so far) and either emits :class:`ToolCall` objects or
a final answer. It is deterministic, so every trace in the lessons is
reproducible. It is *not* intelligent: change the wording of a request and it
may do something silly, which is a useful reminder of what real models add.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from .kb import tokenize
from .providers import Completion, FakeProvider, Message, ToolCall, count_tokens

PRODUCT_NAMES = {
    "ember t3": "ember-t3", "glow a19": "glow-a19", "spark mini": "spark-mini",
    "watcher s1": "watcher-s1", "lookout c2": "lookout-c2", "hearth hub": "hearth-hub",
}
ORDER_RE = re.compile(r"\b((?:HW-)?(\d{6}))\b", re.IGNORECASE)
POLICY_WORDS = {"policy", "refund", "shipping", "delivery", "warranty", "plus", "subscription", "restocking"}


def _call_tokens(calls: Sequence[ToolCall]) -> int:
    return sum(count_tokens(c.name) + count_tokens(json.dumps(c.arguments)) for c in calls)


def make_completion(
    messages: Sequence[Message], text: str = "", calls: Sequence[ToolCall] = (),
    tools: Sequence[dict[str, Any]] | None = None, model: str = "scripted-tools-1",
) -> Completion:
    """Build a Completion with honest token counts: messages, earlier tool calls and tool specs all cost."""
    prompt = sum(count_tokens(m.content) + 4 + _call_tokens(m.tool_calls) for m in messages)
    prompt += count_tokens(json.dumps(list(tools))) if tools else 0
    return Completion(text, model, "fake", prompt, count_tokens(text) + _call_tokens(calls), 0.0,
                      tool_calls=tuple(calls), finish_reason="tool_calls" if calls else "stop")


def products_in(text: str) -> list[str]:
    """Product slugs mentioned in ``text``, in order of appearance."""
    low = text.lower()
    found = sorted((low.find(name), slug) for name, slug in PRODUCT_NAMES.items() if name in low)
    return [slug for _, slug in found]


class ScriptedToolModel:
    """Decides the next step from the request and the tool results seen so far."""

    def __init__(self, tool_specs: Sequence[dict[str, Any]]) -> None:
        self.tool_specs = list(tool_specs)
        self.available = {s["function"]["name"] for s in self.tool_specs}

    def __call__(self, messages: Sequence[Message]) -> Completion:
        user = next(m.content for m in messages if m.role == "user")
        seen = [(m.name, json.loads(m.content)) for m in messages if m.role == "tool"]
        turn = sum(1 for m in messages if m.role == "assistant" and m.tool_calls)
        decision = self.decide(user, seen)
        if isinstance(decision, str):
            return make_completion(messages, text=decision, tools=self.tool_specs)
        calls = [ToolCall(f"call_{turn}_{i}", name, args) for i, (name, args) in enumerate(decision)]
        return make_completion(messages, calls=calls, tools=self.tool_specs)

    def decide(self, user: str, seen: list[tuple[str | None, dict[str, Any]]]) -> str | list[tuple[str, dict]]:
        low = user.lower()
        last = {name: payload for name, payload in seen}  # latest result per tool
        products = products_in(user)
        if "compare" in low and len(products) >= 2:
            if "get_product" not in last:
                return [("get_product", {"product": p}) for p in products]  # parallel calls
            facts = [p["data"] for _, p in seen if p["ok"]]
            return " ".join(f"{f['name']}: €{f['price_eur']:g}, Wi-Fi {f['wifi']}, "
                            f"needs hub: {'yes' if f['needs_hub'] else 'no'}." for f in facts)
        match = ORDER_RE.search(user)
        if match:
            lookups = sum(1 for name, _ in seen if name == "lookup_order")
            return self._order_flow(low, match, last, products, lookups)
        if POLICY_WORDS & set(tokenize(user)) or "return" in low:
            if "search_help_centre" not in last:
                return [("search_help_centre", {"query": " ".join(tokenize(user)[:6])})]
            hits = last["search_help_centre"].get("data") or []
            if not hits:
                return "I could not find that in the help centre. Let me pass you to a colleague."
            return f"{hits[0]['snippet']} [{hits[0]['id']}]"
        return "I can help with orders, returns, products and Hearthwise policies. What do you need?"

    def _order_flow(
        self, low: str, match: re.Match[str], last: dict[str, Any], products: list[str], lookups: int
    ) -> str | list[tuple[str, dict]]:
        typed, digits = match.group(1), match.group(2)
        lookup = last.get("lookup_order")
        if lookup is None:
            return [("lookup_order", {"order_id": typed})]  # copies what the customer typed
        if not lookup["ok"]:
            if lookup["error"]["type"] == "invalid_arguments" and lookups == 1:
                return [("lookup_order", {"order_id": f"HW-{digits}"})]  # self-correction from the error
            return f"I could not find order HW-{digits} in your account. Could you check the number?"
        order = lookup["data"]
        wants_return = "return" in low or "send back" in low
        status = f"Order {order['order_id']} is {order['status']}"
        if order["delivered_on"]:
            status += f" (delivered on {order['delivered_on']})"
        if not wants_return:
            return status + "."
        check = order["return"]
        if not check["eligible"]:
            return f"{status}. It cannot be returned: {check['reason']}."
        if "create_return" not in self.available:
            return (f"{status}. You can still return it ({check['days_left']} days left): open the app, "
                    "go to Account > Orders and tap Return an item.")
        made = last.get("create_return")
        if made is None:
            wanted = {p.replace("-", "") for p in products}
            item = next((i for i in order["items"] if i["name"].lower().replace(" ", "") in wanted),
                        order["items"][0])
            return [("create_return", {"order_id": order["order_id"], "sku": item["sku"], "quantity": 1,
                                       "reason": "changed_mind"})]
        if made["ok"]:
            r = made["data"]
            return (f"Done: return {r['rma']} is open for 1 x {r['sku']}. You will get €{r['refund_eur']:g} "
                    "back within 5 to 7 business days after it reaches our warehouse.")
        if made["error"]["type"] == "rejected":
            return "I have not started the return because it was not approved. A colleague will contact you."
        return f"I could not start the return: {made['error']['message']}"


def scripted_provider(tool_specs: Sequence[dict[str, Any]]) -> FakeProvider:
    """A FakeProvider whose responder is the scripted tool-calling model."""
    return FakeProvider(responder=ScriptedToolModel(tool_specs), model="scripted-tools-1")
