"""Verifiable reasoning problems: every answer is computed, never typed by hand.

Four families:
- ``order_total``   multi-step shop arithmetic with a discount and a shipping rule
- ``energy``        savings and payback calculations
- ``hub_capacity``  counting with distractors (some devices do not use the hub)
- ``target``        reach a target number with + - * / using each number once
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction
from typing import Any

from .catalog import ELECTRICITY_EUR_KWH_2025, PRODUCTS, SEED, SHIPPING

PRICES = {p["name"]: p["price_eur"] for p in PRODUCTS}


def fmt(x: Fraction | float | int) -> str:
    """Canonical answer format: integers without decimals, otherwise two decimals."""
    value = Fraction(x).limit_denominator(10_000) if not isinstance(x, Fraction) else x
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.2f}"


def order_total(rng: random.Random, idx: int) -> dict[str, Any]:
    items = rng.sample(sorted(PRICES), rng.choice([1, 2, 2, 3]))
    qty = {name: rng.randint(1, 4) for name in items}
    discount = rng.choice([0, 10, 15, 20])
    lines = [f"{qty[n]} x {n} at €{PRICES[n]} each" for n in items]
    subtotal = sum(Fraction(PRICES[n]) * qty[n] for n in items)
    steps = [f"Subtotal: " + " + ".join(f"{qty[n]} x {PRICES[n]}" for n in items) + f" = {fmt(subtotal)}"]
    after = subtotal
    if discount:
        after = subtotal * Fraction(100 - discount, 100)
        steps.append(f"Apply {discount}% discount: {fmt(subtotal)} x {(100 - discount) / 100:.2f} = {fmt(after)}")
    free = after > SHIPPING["free_threshold_eur"]
    ship = Fraction(0) if free else Fraction(str(SHIPPING["standard_fee_eur"]))
    steps.append(
        f"Shipping: {fmt(after)} is {'over' if free else 'not over'} €{SHIPPING['free_threshold_eur']}, "
        f"so shipping is {fmt(ship)}"
    )
    total = after + ship
    steps.append(f"Total: {fmt(after)} + {fmt(ship)} = {fmt(total)}")
    disc_txt = f" A {discount}% discount code applies to the products (not to shipping)." if discount else ""
    q = (f"A customer buys {', '.join(lines)}.{disc_txt} Shipping is free when the products cost more than "
         f"€{SHIPPING['free_threshold_eur']} after any discount; otherwise it costs €{SHIPPING['standard_fee_eur']}. "
         "What is the total in euros?")
    return {"id": f"r{idx:03d}", "type": "order_total", "question": q, "answer": fmt(total),
            "answer_type": "number", "steps": steps}


def energy(rng: random.Random, idx: int) -> dict[str, Any]:
    country = rng.choice(sorted(ELECTRICITY_EUR_KWH_2025))
    price = Fraction(str(ELECTRICITY_EUR_KWH_2025[country]))
    kwh = rng.randrange(800, 2500, 100)
    pct = rng.choice([5, 8, 10, 12, 15])
    saved_kwh = Fraction(kwh * pct, 100)
    saved_eur = saved_kwh * price
    steps = [f"Energy saved: {kwh} x {pct}% = {fmt(saved_kwh)} kWh",
             f"Money saved: {fmt(saved_kwh)} x {float(price):.2f} = {fmt(saved_eur)} euros per year"]
    if rng.random() < 0.5:
        q = (f"A home in {country} uses {kwh} kWh of electricity per year for heating. A smart thermostat cuts "
             f"that by {pct}%. Electricity costs €{float(price):.2f} per kWh. How many euros does the home save "
             "per year? Round to two decimals.")
        ans = fmt(saved_eur)
        return {"id": f"r{idx:03d}", "type": "energy", "question": q, "answer": ans, "answer_type": "number",
                "steps": steps}
    cost = 149
    years = Fraction(cost) / saved_eur
    rounded = round(float(years), 1)
    steps.append(f"Payback: {cost} / {fmt(saved_eur)} = {float(years):.3f} years, rounded to {rounded}")
    q = (f"A home in {country} uses {kwh} kWh of electricity per year for heating. An Ember T3 thermostat "
         f"(€{cost}) cuts that by {pct}%. Electricity costs €{float(price):.2f} per kWh. After how many years "
         "does the thermostat pay for itself? Round to one decimal.")
    return {"id": f"r{idx:03d}", "type": "energy", "question": q, "answer": f"{rounded:.1f}",
            "answer_type": "number", "steps": steps}


def hub_capacity(rng: random.Random, idx: int) -> dict[str, Any]:
    floors = rng.randint(1, 3)
    rooms = rng.randint(2, 4)
    motion = rng.randint(1, 2)
    doors = rng.randint(0, 2)
    bulbs = rng.randint(2, 6)
    cameras = rng.randint(0, 3)
    per_room = motion + doors
    used = floors * rooms * per_room
    capacity = 64
    steps = [
        f"HomeMesh devices per room: {motion} motion + {doors} door = {per_room}",
        f"Rooms: {floors} floors x {rooms} rooms = {floors * rooms}",
        f"HomeMesh devices: {floors * rooms} x {per_room} = {used}",
        "Bulbs and cameras use Wi-Fi, so they do not count",
        f"Free slots: {capacity} - {used} = {capacity - used}",
    ]
    q = (f"A Hearth Hub can connect up to {capacity} HomeMesh devices. Motion sensors and door sensors use "
         "HomeMesh; bulbs and cameras use Wi-Fi and do not count. A house has {floors} floors with {rooms} rooms "
         "each. Every room has {motion} motion sensor(s), {doors} door sensor(s) and {bulbs} bulbs, and the house "
         "has {cameras} cameras. How many free HomeMesh slots are left on the hub?").format(
        floors=floors, rooms=rooms, motion=motion, doors=doors, bulbs=bulbs, cameras=cameras)
    return {"id": f"r{idx:03d}", "type": "hub_capacity", "question": q, "answer": str(capacity - used),
            "answer_type": "number", "steps": steps}


OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else None,
}


def _solutions(nums: list[int]) -> dict[int, tuple[str, list[str]]]:
    """Enumerate expressions over ``nums``; map each positive integer result to its first solution."""
    found: dict[int, tuple[str, list[str]]] = {}

    def combine(items: list[tuple[Fraction, str, list[str]]]) -> None:
        if len(items) == 1:
            value, expr, steps = items[0]
            if value.denominator == 1 and 1 <= value <= 200 and int(value) not in found:
                found[int(value)] = (expr, steps)
            return
        for i, j in itertools.permutations(range(len(items)), 2):
            rest = [items[k] for k in range(len(items)) if k not in (i, j)]
            (va, ea, sa), (vb, eb, sb) = items[i], items[j]
            for sym, fn in OPS.items():
                out = fn(va, vb)
                if out is None or out < 0:
                    continue
                step = f"{fmt(va)} {sym} {fmt(vb)} = {fmt(out)}"
                combine(rest + [(out, f"({ea} {sym} {eb})", sa + sb + [step])])

    combine([(Fraction(n), str(n), []) for n in nums])
    return found


def target(rng: random.Random, idx: int) -> dict[str, Any]:
    k = rng.choice([3, 4, 4])
    while True:
        nums = sorted(rng.randint(1, 13) for _ in range(k))
        sols = _solutions(nums)
        candidates = sorted(t for t in sols if 10 <= t <= 100)
        if candidates:
            break
    goal = rng.choice(candidates)
    expr, steps = sols[goal]
    expr = expr[1:-1] if expr.startswith("(") and expr.endswith(")") else expr
    q = (f"Using each of the numbers {', '.join(map(str, nums))} exactly once, and only +, -, * and /, write an "
         f"expression equal to {goal}.")
    return {"id": f"r{idx:03d}", "type": "target", "question": q, "answer": expr, "answer_type": "expression",
            "numbers": nums, "target": goal, "steps": steps}


def build_reasoning() -> list[dict[str, Any]]:
    rng = random.Random(f"{SEED}-reasoning")
    rows: list[dict[str, Any]] = []
    plan = [order_total] * 40 + [energy] * 35 + [hub_capacity] * 35 + [target] * 50
    for i, fn in enumerate(plan, 1):
        row = fn(rng, i)
        row["difficulty"] = len(row["steps"])
        rows.append(row)
    return rows
