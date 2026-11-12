"""The dataset loader, the exact verifiers and the arithmetic re-check."""

from __future__ import annotations

from collections import Counter

from m09_reasoning.problems import check_answer, load_problems, verify_expression
from m09_reasoning.reasoner import Attempt
from m09_reasoning.verifiers import SimulatedORM, SimulatedPRM, first_bad_step, step_is_consistent


def by_id(pid: str):
    return next(p for p in load_problems() if p.id == pid)


def test_dataset_shape() -> None:
    problems = load_problems()
    assert len(problems) == 160
    assert Counter(p.type for p in problems) == {
        "order_total": 40,
        "energy": 35,
        "hub_capacity": 35,
        "target": 50,
    }
    assert all(p.difficulty == len(p.steps) for p in problems)


def test_every_target_reference_passes_the_verifier() -> None:
    targets = [p for p in load_problems() if p.type == "target"]
    assert (
        sum(verify_expression(p.numbers, p.target, p.answer) for p in targets) == 50
    )  # datasets/README baseline


def test_expression_verifier_rejects_cheating() -> None:
    p = by_id("r111")  # 8, 9, 9, 13 -> 50
    assert verify_expression(p.numbers, 50, "(8 * 9) - (9 + 13)")
    assert verify_expression(p.numbers, 50, "8 * 9 - (13 + 9)")
    assert not verify_expression(p.numbers, 50, "(8 * 9) - 22")  # does not use each number
    assert not verify_expression(p.numbers, 50, "8 * 9 - 9 - 13 + 0")  # extra number
    assert not verify_expression(p.numbers, 51, "(8 * 9) - (9 + 13)")  # wrong value
    assert not verify_expression(p.numbers, 50, "__import__('os')")
    assert not verify_expression(p.numbers, 50, "8 ** 9")
    assert not verify_expression(p.numbers, 50, None)


def test_numeric_answers_compare_as_numbers() -> None:
    p = by_id("r001")  # 175.10
    assert check_answer(p, "175.1") and check_answer(p, "€175.10") and not check_answer(p, "175.2")
    assert not check_answer(p, None)


def test_recheck_accepts_every_reference_step() -> None:
    steps = [s for p in load_problems() for s in p.steps]
    assert len(steps) == 542
    assert all(step_is_consistent(s) for s in steps)


def test_recheck_catches_slips_but_not_misreadings() -> None:
    assert not step_is_consistent("Subtotal: 3 x 19 + 1 x 149 = 216")
    assert not step_is_consistent("Shipping: 30.90 is over €49, so shipping is 0")
    assert not step_is_consistent("Payback: 149 / 32.30 = 4.613 years, rounded to 4.7")
    assert not step_is_consistent("Energy saved: 2300 x 12% = 286 kWh")
    # Counting the bulbs is wrong, but the arithmetic is fine: the re-check cannot see it.
    assert step_is_consistent("HomeMesh devices per room: 1 motion + 1 door + 5 bulbs = 7")
    assert first_bad_step(["Rooms: 2 floors x 3 rooms = 6", "HomeMesh devices: 6 x 7 = 52"]) == 2


def test_simulated_reward_models_are_deterministic_and_informative() -> None:
    p = by_id("r076")
    good = Attempt("good", p.steps, p.answer, 40)
    bad_steps = ("HomeMesh devices per room: 1 motion + 1 door + 5 bulbs = 7", *p.steps[1:])
    orm, prm = SimulatedORM(), SimulatedPRM()
    assert orm.score(p, good) == orm.score(p, good)
    assert prm.step_scores(p, p.steps) == prm.step_scores(p, p.steps)
    assert prm.score(p, p.steps) > prm.score(p, bad_steps)
    longer = SimulatedORM(length_bias=5.0)
    assert longer.score(p, Attempt("bad", bad_steps, "22", 200)) > longer.score(
        p, Attempt("bad", bad_steps, "22", 20)
    )
