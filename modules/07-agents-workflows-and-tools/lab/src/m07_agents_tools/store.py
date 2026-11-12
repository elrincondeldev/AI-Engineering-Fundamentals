"""A small, seeded, in-memory order store (synthetic data generated inside the lab).

Five demo orders are fixed so the lessons can refer to them; the rest are
generated with ``random.Random(seed)``. Prices come from the published product
pages, and no order contains a Lookout C2 before its launch on 2025-09-01.

The lab's "today" is fixed (:data:`TODAY`) so return-window checks never change.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from .kb import PRODUCT_SLUGS, get_product

TODAY = date(2025, 11, 14)
RETURN_WINDOW_DAYS = 30  # returns policy 2025 (kb-011)
LOOKOUT_LAUNCH = date(2025, 9, 1)
DEMO_CUSTOMER = "C-0042"


class ToolError(Exception):
    """A business-rule failure the model should read and explain (not retryable)."""


class NotFoundError(ToolError):
    """The record does not exist *for this customer* (we never reveal which one)."""


class TransientToolError(Exception):
    """A temporary failure (a busy database, a dropped connection): safe to retry."""


@dataclass(frozen=True)
class OrderLine:
    sku: str
    name: str
    quantity: int
    unit_price_eur: float


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    placed_on: date
    status: str  # processing | shipped | delivered | cancelled
    delivered_on: date | None
    lines: tuple[OrderLine, ...]

    @property
    def total_eur(self) -> float:
        return round(sum(line.quantity * line.unit_price_eur for line in self.lines), 2)


def _line(slug: str, quantity: int) -> OrderLine:
    p = get_product(slug)
    return OrderLine(str(p["sku"]), str(p["name"]), quantity, float(p["price_eur"]))  # type: ignore[arg-type]


def _demo_orders() -> list[Order]:
    d = date
    return [
        Order("HW-310042", DEMO_CUSTOMER, d(2025, 10, 20), "delivered", d(2025, 10, 23), (_line("ember-t3", 1),)),
        Order("HW-310077", DEMO_CUSTOMER, d(2025, 9, 22), "delivered", d(2025, 9, 26), (_line("glow-a19", 4),)),
        Order("HW-310108", DEMO_CUSTOMER, d(2025, 11, 11), "shipped", None, (_line("lookout-c2", 1),)),
        Order("HW-310155", DEMO_CUSTOMER, d(2025, 10, 31), "delivered", d(2025, 11, 3),
              (_line("spark-mini", 2), _line("hearth-hub", 1))),
        Order("HW-418230", "C-0107", d(2025, 10, 28), "delivered", d(2025, 11, 1),
              (_line("watcher-s1", 2), _line("hearth-hub", 1))),
    ]


def generate_orders(seed: int = 7, n_random: int = 55) -> list[Order]:
    """Fixed demo orders plus ``n_random`` seeded random ones."""
    rng = random.Random(seed)
    orders = _demo_orders()
    used = {o.order_id for o in orders}
    start = date(2025, 8, 1)
    span = (TODAY - start).days
    while len(orders) < len(_demo_orders()) + n_random:
        oid = f"HW-{rng.randint(500000, 999999)}"
        if oid in used:
            continue
        used.add(oid)
        placed = start + timedelta(days=rng.randrange(span))
        catalogue = [s for s in PRODUCT_SLUGS if s != "lookout-c2" or placed >= LOOKOUT_LAUNCH]
        slugs = rng.sample(catalogue, k=rng.choice([1, 1, 2, 3]))
        lines = tuple(_line(s, rng.choice([1, 1, 2])) for s in slugs)
        age = (TODAY - placed).days
        if rng.random() < 0.05:
            status, delivered = "cancelled", None
        elif age <= 2:
            status, delivered = "processing", None
        elif age <= 5:
            status, delivered = "shipped", None
        else:
            status, delivered = "delivered", placed + timedelta(days=rng.randint(2, 4))
        orders.append(Order(oid, f"C-{rng.randint(1, 200):04d}", placed, status, delivered, lines))
    return orders


@dataclass
class OrderStore:
    """Orders plus the returns created during a session."""

    seed: int = 7
    orders: dict[str, Order] = field(init=False)
    returns: list[dict[str, object]] = field(default_factory=list, init=False)
    _by_key: dict[str, dict[str, object]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.orders = {o.order_id: o for o in generate_orders(self.seed)}

    def _owned(self, order_id: str, customer_id: str) -> Order:
        order = self.orders.get(order_id)
        if order is None or order.customer_id != customer_id:
            raise NotFoundError(f"No order {order_id} found for this customer.")
        return order

    @staticmethod
    def eligibility(order: Order) -> dict[str, object]:
        """Return-window check against the 2025 policy (30 days from delivery)."""
        if order.status != "delivered" or order.delivered_on is None:
            return {"eligible": False, "reason": f"order is {order.status}, not delivered yet"}
        days = (TODAY - order.delivered_on).days
        if days > RETURN_WINDOW_DAYS:
            return {"eligible": False, "reason": f"delivered {days} days ago; the window is "
                    f"{RETURN_WINDOW_DAYS} days", "days_since_delivery": days}
        return {"eligible": True, "days_since_delivery": days, "days_left": RETURN_WINDOW_DAYS - days}

    def lookup(self, order_id: str, customer_id: str) -> dict[str, object]:
        order = self._owned(order_id, customer_id)
        return {
            "order_id": order.order_id,
            "status": order.status,
            "placed_on": order.placed_on.isoformat(),
            "delivered_on": order.delivered_on.isoformat() if order.delivered_on else None,
            "items": [{"sku": ln.sku, "name": ln.name, "quantity": ln.quantity} for ln in order.lines],
            "total_eur": order.total_eur,
            "return": self.eligibility(order),
        }

    def create_return(
        self, order_id: str, customer_id: str, sku: str, quantity: int, reason: str,
        idempotency_key: str | None = None,
    ) -> dict[str, object]:
        """Create a return authorisation (RMA). With a key, repeating the call is harmless."""
        if idempotency_key and idempotency_key in self._by_key:
            return {**self._by_key[idempotency_key], "duplicate": True}
        order = self._owned(order_id, customer_id)
        check = self.eligibility(order)
        if not check["eligible"]:
            raise ToolError(f"Order {order_id} cannot be returned: {check['reason']}.")
        line = next((ln for ln in order.lines if ln.sku == sku), None)
        if line is None:
            raise ToolError(f"Order {order_id} does not contain {sku}.")
        already = sum(int(r["quantity"]) for r in self.returns if r["order_id"] == order_id and r["sku"] == sku)
        if quantity > line.quantity - already:
            raise ToolError(f"Only {line.quantity - already} x {sku} can still be returned.")
        seed = idempotency_key or f"{order_id}:{sku}:{len(self.returns)}"
        record: dict[str, object] = {
            "rma": "RMA-" + hashlib.sha256(seed.encode()).hexdigest()[:6].upper(),
            "order_id": order_id, "sku": sku, "quantity": quantity, "reason": reason,
            "refund_eur": round(quantity * line.unit_price_eur, 2),
        }
        self.returns.append(record)
        if idempotency_key:
            self._by_key[idempotency_key] = record
        return {**record, "duplicate": False}
