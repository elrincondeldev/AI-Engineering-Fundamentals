"""Context assembly under a budget, compaction and prefix caching."""

from __future__ import annotations

from m04_adaptation.context import ContextAssembler, ContextItem, PrefixCache, drop_near_duplicates
from m04_adaptation.providers import Message


def _items() -> list[ContextItem]:
    return [
        ContextItem("system", "You route tickets. " * 10, priority=100, required=True, stable=True),
        ContextItem("example", "Ticket: my parcel is late", priority=0.9, answer="order_status"),
        ContextItem("example", "Ticket: my parcel is late!", priority=0.8, answer="order_status"),
        ContextItem("example", "Ticket: invoice with wrong VAT number please", priority=0.2, answer="billing"),
        ContextItem("profile", "Customer context: language=en", priority=50),
        ContextItem("query", "Ticket: where is my parcel", priority=1000, required=True),
    ]


def test_near_duplicates_are_dropped() -> None:
    kept, dropped = drop_near_duplicates(_items())
    assert [d.text for d in dropped] == ["Ticket: my parcel is late!"]
    assert len(kept) == 5


def test_budget_is_respected_and_required_items_survive() -> None:
    items = _items()
    required = sum(i.tokens for i in items if i.required)
    a = ContextAssembler(budget=required + 12, dedupe=True).assemble(items)
    assert a.tokens <= required + 12
    kinds = [i.kind for i in a.kept]
    assert "system" in kinds and "query" in kinds and "profile" in kinds
    assert a.examples_kept == 0
    reasons = {why for _, why in a.dropped}
    assert reasons == {"near-duplicate", "over budget"}


def test_order_puts_stable_prefix_first_and_query_last() -> None:
    a = ContextAssembler(budget=10_000).assemble(_items())
    assert a.messages[0].role == "system"
    assert a.messages[-1].content.endswith("Ticket: where is my parcel")
    assert a.messages[-1].content.startswith("Customer context:")
    examples = [m.content for m in a.messages if m.role == "user"][:-1]
    assert examples[-1] == "Ticket: my parcel is late"  # most relevant example closest to the query


def test_prefix_cache_counts_hits() -> None:
    cache = PrefixCache()
    for q in ("a", "b", "c"):
        cache.observe([Message("system", "same prefix"), Message("user", q)])
    assert (cache.hits, cache.misses) == (2, 1)
    assert cache.hit_rate == 0.667
