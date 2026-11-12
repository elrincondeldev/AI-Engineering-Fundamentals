"""The scripted model's skills: reading questions, extracting facts, resolving conflicts."""

from __future__ import annotations

import pytest

from m08_web_agent.skills import Fact, analyze, compose, extract, find_entities, parse_number, resolve


@pytest.mark.parametrize(
    ("question", "op", "attributes"),
    [
        ("How many devices can a single HomeMesh 2.0 network support?", "lookup", ["devices"]),
        (
            "How much more expensive is the Corvid Thermo X than the Hearthwise Ember T3, in euros?",
            "difference",
            ["price", "price"],
        ),
        (
            "Which bulb gives more lumens per euro: the Glow A19 or the Brightline Bulb Pro?",
            "per_unit_max",
            ["lumens", "price", "lumens", "price"],
        ),
        (
            "Does the Corvid Thermo X meet the EHEO standby power recommendation after its firmware update?",
            "within_limit",
            ["standby", "max_standby"],
        ),
        ("What is the price of the Corvid Thermo X in Mexico, in Mexican pesos?", "lookup", ["price_mxn"]),
    ],
)
def test_analyze_question_shapes(question: str, op: str, attributes: list[str]) -> None:
    plan = analyze(question)
    assert plan.op == op
    assert [s.attribute for s in plan.slots] == attributes


def test_shared_query_for_country_comparisons() -> None:
    plan = analyze("Which country has the higher smart home adoption rate in 2025: Mexico or Chile?")
    assert plan.options == ("Mexico", "Chile")
    assert len({s.query for s in plan.slots}) == 1


def test_entities_and_numbers() -> None:
    assert "Corvid Thermo X" in find_entities("Does the Corvid Thermo X meet the EHEO recommendation?")
    assert parse_number("410,000") == parse_number("410.000") == 410000
    assert parse_number("2 million") == 2_000_000
    assert parse_number("0.4 W") == 0.4


def test_extract_respects_entity_and_attribute() -> None:
    plan = analyze("By what percentage did smart thermostat sales grow in Colombia in 2025?")
    text = "Spain: 410,000 units (6% growth); Colombia: 120,000 units (22% growth)."
    fact = extract(plan.slots[0], text, "https://marketlens.example/r", "2025-10-15", "high")
    assert fact is not None and fact.value == "22%"
    assert extract(plan.slots[0], "Colombia is lovely.", "https://x.example/a", "", "") is None


def _f(value: str, tier: str, published: str, order: int) -> Fact:
    return Fact("s", value, f"https://{order}.example/", published, tier, "", order)


def test_resolve_rules_change_the_winner() -> None:
    spec, forum, copy = (
        _f("250", "high", "2024-04-16", 0),
        _f("200", "low", "2024-08-01", 1),
        _f("200", "low", "2024-08-01", 2),
    )
    assert resolve([spec, forum], prefer_credible=False, prefer_recent=False).value == "250"  # first seen
    assert resolve([spec, forum, copy], prefer_credible=False, prefer_recent=False).value == "200"  # votes
    assert resolve([spec, forum], prefer_credible=False, prefer_recent=True).value == "200"  # newest
    assert resolve([spec, forum, copy], prefer_credible=True, prefer_recent=True).value == "250"  # tier first


def test_compose_unknown_when_a_fact_is_missing() -> None:
    plan = analyze("What is the price of the Corvid Thermo X in Mexico, in Mexican pesos?")
    answer = compose(plan, {plan.slots[0].key: None})
    assert answer.short == "unknown" and answer.sources == ()
