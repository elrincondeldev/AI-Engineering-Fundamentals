"""Query decomposition: turn one research question into focused search sub-queries.

The planner asks the LLM provider for JSON. With the default FakeProvider, the
"model" is :func:`fake_planner`, a rule-based stand-in that recognises the
entities and the intent of the question. A real model writes the sub-queries
itself; the parsing and validation code is the same for both.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .providers import LLMProvider, Message

COUNTRIES = ["Spain", "Portugal", "Mexico", "Colombia", "Chile"]
PRODUCTS = {  # product name -> category, for Hearthwise and known competitors
    "Ember T3": "smart thermostat",
    "Glow A19": "smart bulb",
    "Spark Mini": "smart plug",
    "Watcher S1": "motion sensor",
    "Lookout C2": "indoor camera",
    "Hearth Hub": "smart home hub",
    "Corvid Thermo X": "smart thermostat",
    "Brightline Bulb Pro": "smart bulb",
    "Tessera Cam 4": "indoor camera",
}

PLANNER_SYSTEM = (
    "You are the planner of a research agent. Split the question into 2 to 6 web-search "
    "sub-queries that together cover what is needed to answer it. Name the entities each "
    'sub-query is about. Reply with JSON only: {"subqueries": [{"aspect": "...", '
    '"query": "...", "entities": ["..."]}]}'
)


@dataclass(frozen=True)
class SubQuery:
    id: str
    aspect: str
    query: str
    entities: tuple[str, ...]


def found_entities(question: str) -> tuple[list[str], list[str]]:
    """Return (countries, products) named in ``question``, in order of appearance."""
    def ordered(names: list[str]) -> list[str]:
        hits = [(question.find(n), n) for n in names if n in question]
        return [n for _, n in sorted(hits)]

    return ordered(COUNTRIES), ordered(list(PRODUCTS))


def fake_planner(messages: Sequence[Message]) -> str:
    """Rule-based planner used as the FakeProvider responder (deterministic)."""
    question = messages[-1].content.removeprefix("Question:").strip()
    q = question.lower()
    countries, products = found_entities(question)
    where = " ".join(countries)
    category = PRODUCTS[products[0]] if products else "smart home device"
    subs: list[dict] = []

    def add(aspect: str, query: str, entities: list[str]) -> None:
        subs.append({"aspect": aspect, "query": query, "entities": entities})

    if re.search(r"\b(launch|market|enter)\b", q):  # market entry: demand, costs, logistics, rivals
        add("market", f"{category} sales 2025 growth {where}", countries)
        add("adoption", f"smart home adoption households {where}", countries)
        add("energy prices", f"household electricity price kWh {where}", countries)
        add("logistics", f"parcel delivery time days {where}", countries)
        add("competition", f"competitor {category} available {where}", countries)
    if re.search(r"\b(compare|versus|vs)\b", q):  # comparison: one query per product and attribute
        for p in products:
            add("price", f"{p} price", [p])
            if "standby" in q:
                add("standby power", f"{p} standby power", [p])
        if "guideline" in q:
            add("guidelines", "recommended maximum standby power connected devices guideline", [])
    if re.search(r"\b(evidence|save|savings)\b", q):  # evidence: studies and the claims people repeat
        add("studies", "smart thermostat heating energy savings study homes", [])
        add("claims", "how much do smart thermostats save", [])
    if not subs:
        add("general", question, countries + products)
    return json.dumps({"subqueries": subs})


def parse_plan(text: str, question: str, max_subqueries: int = 6) -> list[SubQuery]:
    """Validate the planner's JSON. On any problem fall back to one sub-query: the question."""
    match = re.search(r"\{.*\}", text, re.S)
    try:
        raw = json.loads(match.group(0))["subqueries"] if match else []
        subs = [
            SubQuery(f"s{i}", str(r.get("aspect", "general")), str(r["query"]),
                     tuple(str(e) for e in r.get("entities", [])))
            for i, r in enumerate(raw[:max_subqueries], 1)
            if isinstance(r, dict) and str(r.get("query", "")).strip()
        ]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        subs = []
    return subs or [SubQuery("s1", "general", question, ())]


def plan(question: str, provider: LLMProvider, max_subqueries: int = 6) -> list[SubQuery]:
    """Ask the provider for a plan and return validated sub-queries."""
    out = provider.complete(
        [Message("system", PLANNER_SYSTEM), Message("user", f"Question: {question}")],
        temperature=0.0,
        max_tokens=400,
        seed=0,
    )
    return parse_plan(out.text, question, max_subqueries)
