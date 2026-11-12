"""Loading, cleaning, deduplication and splits."""

from __future__ import annotations

from m04_adaptation.tickets import (
    LAUNCH_DATE,
    clean_text,
    deduplicate,
    find_resubmissions,
    four_way_split,
    load_tickets,
    random_split,
    redact_pii,
    temporal_split,
)


def test_load_and_deduplicate() -> None:
    rows = load_tickets()
    assert len(rows) == 1224
    pairs = find_resubmissions(rows)
    assert len(pairs) == 24
    assert all(a.body == b.body and a.customer_email == b.customer_email for a, b in pairs)
    assert len(deduplicate(rows)) == 1200


def test_clean_text_removes_quotes_signatures_entities_and_pii() -> None:
    raw = ("Hi, my Ember T3 won&#39;t connect.\n\n> On 2025-01-01, Hearthwise Support wrote:\n> Share your order"
           "\n\nBest regards\nLaura Gómez\n+34 612 345 678")
    assert clean_text(raw) == "Hi, my Ember T3 won't connect."
    assert clean_text("Call me on +34 612 345 678, please.") == "Call me on [PHONE], please."
    assert redact_pii("call +34 612 345 678 or mail a.b@mail.example") == "call [PHONE] or mail [EMAIL]"


def test_temporal_split_never_mixes_past_and_future() -> None:
    train, test = temporal_split(deduplicate(load_tickets()))
    assert max(t.created_at for t in train) < LAUNCH_DATE <= min(t.created_at for t in test)
    assert (len(train), len(test)) == (800, 400)


def test_four_way_split_is_ordered_in_time() -> None:
    s = four_way_split(deduplicate(load_tickets()))
    parts = [s.old_train, s.old_test, s.new_adapt, s.new_test]
    assert [len(p) for p in parts] == [704, 96, 97, 303]
    for earlier, later in zip(parts, parts[1:]):
        assert max(t.created_at for t in earlier) < min(t.created_at for t in later)


def test_random_split_is_deterministic() -> None:
    rows = load_tickets()
    a, b = random_split(rows, seed=0), random_split(rows, seed=0)
    assert [t.ticket_id for t in a[1]] == [t.ticket_id for t in b[1]]
