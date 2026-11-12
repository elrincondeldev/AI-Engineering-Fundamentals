"""The scripted "LLM" behind every workflow step (a ``FakeProvider`` responder).

Each workflow step sends a system prompt that starts with ``TASK: <name>``,
like a different prompt template to the same model. This responder answers
each task with simple, deterministic rules so that the *plumbing* of the
patterns (gates, routes, fan-out, loops) is what you study. A real model would
be smarter and less predictable; the plumbing stays the same.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence

from .providers import Message
from .scripted import ORDER_RE, products_in

LANES = ("orders", "devices", "account")
LANE_KEYWORDS = {
    "orders": ["order", "pedido", "tracking", "shipped", "enviado", "deliver", "arrive", "lleg", "return",
               "devuelv", "devolver", "de vuelta", "send back", "money back", "label", "etiqueta", "bought", "parcel"],
    "devices": ["wifi", "wi-fi", "connect", "conexi", "conect", "pair", "vincul", "offline", "broken",
                "stopped working", "dejó de funcionar", "fault", "defectuos", "warranty", "garantía", "replace",
                "hot", "calienta", "battery", "batería", "compatible", "work with", "outdoors", "factory reset",
                "firmware", "unreachable", "no enciende", "won't", "parpadea", "freezes", "fogged", "dead",
                "murió", "flicker", "difference", "detect", "times out", "range"],
    "account": ["account", "cuenta", "password", "contraseña", "log in", "login", "email address", "charge",
                "cobrad", "statement", "invoice", "factura", "payment", "pago", "paypal", "card", "tarjeta",
                "subscription", "suscripci", "plus", "cancel", "two-factor", "get back in", "bloquead"],
}

STALE_POLICY_ANSWER = (
    "Yes, you can return it within 14 days of delivery. Because the box is opened, a 10% restocking fee "
    "applies to your refund."
)


def strip_quoted(text: str) -> str:
    """Drop quoted earlier replies (lines starting with '>'): they are not the customer's words."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))


def route(text: str) -> str:
    """Keyword router: the lane with most keyword hits; no hits or a tie goes to a human."""
    low = text.lower()
    hits = {lane: sum(k in low for k in words) for lane, words in LANE_KEYWORDS.items()}
    best = max(hits.values())
    winners = [lane for lane, h in hits.items() if h == best]
    return winners[0] if best > 0 and len(winners) == 1 else "human"


def noisy_route(text: str, sample: int, noise: float) -> str:
    """The same router, but each sample is wrong with probability ``noise`` (independently)."""
    clean = route(text)
    digest = hashlib.sha256(f"{sample}|{text}".encode()).hexdigest()
    rng = random.Random(int(digest[:12], 16))
    if rng.random() < noise:
        return rng.choice([lane for lane in LANES if lane != clean])
    return clean


def _evaluate(draft: str) -> dict[str, object]:
    issues = []
    if "14 days" in draft:
        issues.append("states a 14-day window; the current policy (kb-011) says 30 days")
    if "restocking fee" in draft and "no restocking fee" not in draft:
        issues.append("charges a restocking fee; kb-011 says there is none")
    if "[kb-" not in draft:
        issues.append("no citation of the policy document")
    if len(draft.split()) > 70:
        issues.append("longer than 70 words")
    return {"pass": not issues, "issues": issues}


def _plan(request: str) -> list[str]:
    low = request.lower()
    wanted = [("light", "glow-a19"), ("heating", "ember-t3"), ("plug", "spark-mini"),
              ("motion", "watcher-s1"), ("camera", "lookout-c2")]
    return [slug for word, slug in wanted if word in low] or products_in(request)


def _replan(payload: dict) -> list[str]:
    """Second look at worker results: a device that needs the hub adds a hub subtask."""
    done = {r["product"] for r in payload["results"]}
    if any(r["needs_hub"] for r in payload["results"]) and "hearth-hub" not in done:
        return ["hearth-hub"]
    return []


def _synthesise(payload: dict) -> str:
    budget = payload["budget_eur"]
    results = payload["results"]
    total = sum(r["price_eur"] for r in results)
    names = ", ".join(f"{r['name']} (€{r['price_eur']:g})" for r in results)
    verdict = "within budget" if total <= budget else f"over budget by €{total - budget:g}"
    return f"Suggested bundle: {names}. Total €{total:g}, {verdict} (€{budget:g})."


def workflow_model(messages: Sequence[Message]) -> str:
    """Answer one workflow step. The task name is on the first line of the system prompt."""
    header = messages[0].content.splitlines()
    task = header[0].removeprefix("TASK: ").strip()
    content = messages[-1].content
    if task == "extract":
        match = ORDER_RE.search(content)
        low = content.lower()
        intent = "return" if ("return" in low or "send" in low) else "status"
        return json.dumps({"order_id": match.group(1) if match else None, "intent": intent})
    if task == "draft_reply":
        facts = json.loads(content)
        order, check = facts["order"], facts["order"]["return"]
        if facts["intent"] == "status":
            return f"Your order {order['order_id']} is {order['status']}."
        if check["eligible"]:
            return (f"Your order {order['order_id']} can be returned: {check['days_left']} days left. "
                    "Open the app, go to Account > Orders and tap Return an item.")
        return f"Sorry, order {order['order_id']} cannot be returned: {check['reason']}."
    if task == "route":
        return route(content)
    if task == "route_noisy":
        sample, noise = int(header[1].split()[1]), float(header[2].split()[1])
        return noisy_route(content, sample, noise)
    if task.startswith("answer_"):
        return f"[{task.removeprefix('answer_')} lane draft]"
    if task == "product_row":
        f = json.loads(content)
        return f"| {f['name']} | €{f['price_eur']:g} | {f['wifi']} | {'yes' if f['needs_hub'] else 'no'} |"
    if task == "aggregate_table":
        return "| Device | Price | Wi-Fi | Needs hub |\n|---|---|---|---|\n" + content
    if task == "draft_policy_answer":
        payload = json.loads(content)
        if not payload.get("feedback"):
            return STALE_POLICY_ANSWER  # first draft "from memory": the superseded 2023 policy
        e = payload["evidence"]
        return (f"Yes. Under our current returns policy you can return any product within 30 days of "
                f"delivery, opened or not, with no restocking fee. Order {e['order_id']} was delivered "
                f"{e['days_since_delivery']} days ago, so you have {e['days_left']} days left. [kb-011]")
    if task == "evaluate":
        return json.dumps(_evaluate(json.loads(content)["draft"]))
    if task == "plan":
        return json.dumps(_plan(content))
    if task == "product_worker":
        f = json.loads(content)
        return json.dumps({"product": f["slug"], "name": f["name"], "price_eur": f["price_eur"],
                           "needs_hub": f["needs_hub"]})
    if task == "replan":
        return json.dumps(_replan(json.loads(content)))
    if task == "synthesise":
        return _synthesise(json.loads(content))
    return f"unknown task {task!r}"


def budget_of(request: str, default: float = 200.0) -> float:
    """Read a euro budget such as '€150' from the request."""
    match = re.search(r"€\s?(\d+)", request)
    return float(match.group(1)) if match else default
