"""The toy in-context learner reads the prompt and nothing else."""

from __future__ import annotations

from m04_adaptation.icl import RETRY_PREFIX, ToyInContextLearner, parse_prompt
from m04_adaptation.providers import FakeProvider, Message

SYSTEM = """Classify the ticket.

Intents:
- order_status: delivery, tracking, parcel
- billing_question: charge, invoice
- account_access

Rules:
- If two problems, pick the action.
Reply with JSON only."""


def test_parse_prompt_finds_labels_examples_and_query() -> None:
    msgs = [
        Message("system", SYSTEM),
        Message("user", "Ticket: I was charged twice"),
        Message("assistant", '{"intent": "billing_question"}'),
        Message("user", "Customer context: language=en\nTicket: where is my parcel?"),
    ]
    p = parse_prompt(msgs)
    assert p.labels == ["order_status", "billing_question", "account_access"]
    assert p.descriptions["account_access"] == ""
    assert p.examples == [("I was charged twice", "billing_question")]
    assert p.query == "where is my parcel?"
    assert p.context == "language=en"
    assert p.json_mode and not p.reasoning and not p.is_retry


def test_zero_shot_uses_descriptions() -> None:
    learner = ToyInContextLearner(slips=False)
    out = learner([Message("system", SYSTEM), Message("user", "Ticket: the invoice shows a wrong charge")])
    assert out == '{"intent": "billing_question"}'


def test_examples_vote_by_similarity() -> None:
    learner = ToyInContextLearner(slips=False)
    msgs = [
        Message("system", SYSTEM),
        Message("user", "Ticket: cannot sign in, password rejected"),
        Message("assistant", '{"intent": "account_access"}'),
        Message("user", "Ticket: password rejected again"),
    ]
    assert learner(msgs) == '{"intent": "account_access"}'


def test_no_evidence_falls_back_to_first_label() -> None:
    learner = ToyInContextLearner(slips=False)
    assert learner([Message("system", SYSTEM), Message("user", "Ticket: zzz")]) == '{"intent": "order_status"}'


def test_retry_never_slips_and_reasoning_ends_with_answer() -> None:
    learner = ToyInContextLearner()
    base = [Message("system", SYSTEM + "\nThink step by step."), Message("user", "Ticket: invoice please")]
    first = learner(base)
    assert first.startswith("Reasoning:") and "\nAnswer: " in first
    retry = base + [Message("assistant", "oops"), Message("user", f"{RETRY_PREFIX} was invalid.")]
    assert parse_prompt(retry).is_retry
    assert learner(retry).endswith('{"intent": "billing_question"}')


def test_responder_plugs_into_fake_provider() -> None:
    provider = FakeProvider(responder=ToyInContextLearner(slips=False))
    out = provider.complete([Message("system", SYSTEM), Message("user", "Ticket: tracking says delivered")])
    assert out.text == '{"intent": "order_status"}'
    assert out.prompt_tokens > 0


def test_position_bias_lowers_middle_examples() -> None:
    learner = ToyInContextLearner(position_bias=0.5)
    assert learner.position_weight(0, 5) == 1.0
    assert learner.position_weight(4, 5) == 1.0
    assert learner.position_weight(2, 5) == 0.5
