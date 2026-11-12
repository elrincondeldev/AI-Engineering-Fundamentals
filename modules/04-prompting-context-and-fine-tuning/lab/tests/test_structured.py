"""Parsing, validation and retries."""

from __future__ import annotations

from m04_adaptation.providers import FakeProvider, Message
from m04_adaptation.structured import classify, parse_answer
from m04_adaptation.tickets import INTENTS


def test_parse_answer_accepts_only_known_labels() -> None:
    assert parse_answer('{"intent": "order_status"}', INTENTS).label == "order_status"
    assert parse_answer('Sure! {"intent": "order_status"}', INTENTS).label == "order_status"
    assert parse_answer('Reasoning: x\nAnswer: {"intent": "account_access"}', INTENTS).label == "account_access"
    assert parse_answer('{"intent": "Order Status"}', INTENTS).error == "'Order Status' is not one of the allowed intents"
    assert parse_answer("no json here", INTENTS).error == "no JSON object found"
    assert parse_answer("billing_question", INTENTS, json_mode=False).label == "billing_question"


def test_classify_retries_once_with_the_error() -> None:
    answers = iter(['{"intent": "Billing"}', '{"intent": "billing_question"}'])
    provider = FakeProvider(responder=lambda msgs: next(answers))
    result = classify(provider, [Message("user", "Ticket: x")], INTENTS)
    assert result.label == "billing_question"
    assert result.attempts == 2 and not result.valid_first_try
    retry_msg = provider.calls[1][-1].content
    assert retry_msg.startswith("Your previous answer was invalid") and "'Billing'" in retry_msg


def test_classify_gives_up_after_max_retries() -> None:
    provider = FakeProvider(responder=lambda msgs: "nope")
    result = classify(provider, [Message("user", "Ticket: x")], INTENTS, max_retries=0)
    assert result.label is None and result.attempts == 1
