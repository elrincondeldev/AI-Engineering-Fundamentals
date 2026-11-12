"""Help-centre reader, search and the seeded order store."""

from __future__ import annotations

import pytest

from m07_agents_tools import kb
from m07_agents_tools.store import (
    DEMO_CUSTOMER,
    LOOKOUT_LAUNCH,
    NotFoundError,
    OrderStore,
    ToolError,
    generate_orders,
)


def test_docs_load_and_access_control() -> None:
    docs = kb.load_docs()
    assert len(docs) == 26
    public_current = kb.searchable_docs()
    assert len(public_current) == 24
    assert all(d.audience == "public" and d.status == "current" for d in public_current)
    assert "kb-041" not in {d.id for d in public_current}  # internal playbook
    assert "kb-010" not in {d.id for d in public_current}  # superseded 2023 policy


def test_hidden_html_is_removed() -> None:
    community = next(d for d in kb.load_docs() if d.id == "kb-025")
    assert community.hidden_removed == 1
    assert "ignore all previous instructions" not in community.text.lower()
    assert sum(d.hidden_removed for d in kb.load_docs()) == 1


def test_search_finds_the_current_returns_policy() -> None:
    hits = kb.search("return window opened item")
    assert hits[0]["id"] == "kb-011"
    assert "30 days" in str(hits[0]["snippet"])
    refund = kb.search("refunds after return")[0]
    assert refund["id"] == "kb-011" and "5 to 7 business days" in str(refund["snippet"])


def test_search_never_returns_internal_documents() -> None:
    assert all(h["id"] != "kb-041" for h in kb.search("goodwill credit voucher escalation"))
    assert kb.search("goodwill credit voucher", include_internal=True)[0]["id"] == "kb-041"


def test_products_match_the_catalogue() -> None:
    prices = {s: kb.get_product(s)["price_eur"] for s in kb.PRODUCT_SLUGS}
    assert prices == {"ember-t3": 149, "glow-a19": 12, "spark-mini": 19, "watcher-s1": 29,
                      "lookout-c2": 89, "hearth-hub": 69}
    assert kb.get_product("watcher-s1")["needs_hub"] is True
    assert kb.get_product("ember-t3")["category"] == "smart thermostat"
    with pytest.raises(KeyError):
        kb.get_product("toaster")


def test_orders_are_deterministic_and_respect_launch_dates() -> None:
    a, b = generate_orders(), generate_orders()
    assert a == b and len(a) == 60
    for order in a:
        if order.placed_on < LOOKOUT_LAUNCH:
            assert all(line.sku != "HW-LKT-C2" for line in order.lines)


def test_lookup_and_ownership() -> None:
    store = OrderStore()
    order = store.lookup("HW-310042", DEMO_CUSTOMER)
    assert order["return"] == {"eligible": True, "days_since_delivery": 22, "days_left": 8}
    with pytest.raises(NotFoundError):
        store.lookup("HW-418230", DEMO_CUSTOMER)  # exists, but belongs to C-0107


def test_return_rules_and_idempotency() -> None:
    store = OrderStore()
    with pytest.raises(ToolError, match="49 days ago"):
        store.create_return("HW-310077", DEMO_CUSTOMER, "HW-GLW-A19", 1, "changed_mind")
    with pytest.raises(ToolError, match="not delivered"):
        store.create_return("HW-310108", DEMO_CUSTOMER, "HW-LKT-C2", 1, "changed_mind")
    first = store.create_return("HW-310155", DEMO_CUSTOMER, "HW-SPK-MINI", 1, "changed_mind", "key-1")
    again = store.create_return("HW-310155", DEMO_CUSTOMER, "HW-SPK-MINI", 1, "changed_mind", "key-1")
    assert again["rma"] == first["rma"] and again["duplicate"] is True
    assert len(store.returns) == 1
    store.create_return("HW-310155", DEMO_CUSTOMER, "HW-SPK-MINI", 1, "changed_mind", "key-2")
    with pytest.raises(ToolError, match="Only 0"):
        store.create_return("HW-310155", DEMO_CUSTOMER, "HW-SPK-MINI", 1, "changed_mind", "key-3")
