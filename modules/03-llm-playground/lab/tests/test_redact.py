"""PII detection, pseudonymisation and the evaluation harness."""

from __future__ import annotations

from m03_playground.redact import Pseudonymiser, detect, gold_spans, ticket_payload

TEXT = "Order #HW-640183, write to ana.ruiz@mail.example or call +34 612 345 678.\nBest regards\nLaura Gómez"


def test_detects_each_kind() -> None:
    kinds = [s.kind for s in detect(TEXT, "names")]
    assert kinds == ["ORDER", "EMAIL", "PHONE", "NAME"]


def test_levels_differ_on_names_and_dates() -> None:
    assert "NAME" not in [s.kind for s in detect(TEXT, "regex")]
    assert [s.text for s in detect("ordered on 2024-12-25", "naive")] == ["2024-12-25"]
    assert detect("ordered on 2024-12-25", "regex") == []


def test_product_names_are_not_people() -> None:
    assert detect("My Spark Mini and the Hearth Hub work fine.", "names") == []


def test_pseudonyms_are_stable_and_reversible() -> None:
    p = Pseudonymiser()
    first, counts = p.redact("Mail ana@mail.example and ana@mail.example")
    assert first == "Mail <EMAIL_1> and <EMAIL_1>"
    assert counts["EMAIL"] == 2
    second, _ = p.redact("Also bo@mail.example")
    assert second == "Also <EMAIL_2>"
    assert p.restore("Sent to <EMAIL_2>") == "Sent to bo@mail.example"


def test_gold_labels_from_the_signature_layout() -> None:
    row = {"customer_email": "c1@mail.example", "body": "Hi.\n\nBest regards\nDiego Torres\n+351 913 456 789"}
    gold = gold_spans(ticket_payload(row))
    assert [(g.kind, g.text) for g in gold] == [
        ("EMAIL", "c1@mail.example"),
        ("NAME", "Diego Torres"),
        ("PHONE", "+351 913 456 789"),
    ]
    payload = ticket_payload(row)
    assert all(payload[g.start : g.end] == g.text for g in gold)
