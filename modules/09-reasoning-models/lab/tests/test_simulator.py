"""The simulated reasoner: faithful to the reference when it does not err, deterministic, honest."""

from __future__ import annotations

import random

import pytest

from m09_reasoning import prompts
from m09_reasoning.problems import check_answer, load_problems
from m09_reasoning.providers import FakeProvider, Message
from m09_reasoning.reasoner import make_reasoner, parse_solution, simulator_of
from m09_reasoning.simulator import SimConfig, SimulatedReasoner, make_family
from m09_reasoning.verifiers import step_is_consistent


def by_id(pid: str):
    return next(p for p in load_problems() if p.id == pid)


def write_all(p, kinds: dict[int, str] | None = None) -> tuple[list[str], str]:
    """Write a whole chain with chosen step kinds (default: all correct)."""
    fam = make_family(p, fudge=0.0)
    chain = fam.start()
    texts = []
    rng = random.Random(0)
    while not fam.done(chain):
        new = chain.copy()
        text, kind = fam.write(new, (kinds or {}).get(chain.phase, "ok"), rng)
        new.phase += 1
        new.kinds = (*new.kinds, kind)
        chain = new
        texts.append(text)
    return texts, fam.answer(chain)


def test_without_errors_the_simulator_reproduces_every_reference_solution() -> None:
    for p in load_problems():
        texts, answer = write_all(p)
        assert tuple(texts) == p.steps, p.id
        assert check_answer(p, answer), p.id


def test_traps_are_tidy_and_slips_are_not() -> None:
    hub = by_id("r076")
    trapped, answer = write_all(hub, {0: "trap"})
    assert trapped[0] == "HomeMesh devices per room: 1 motion + 1 door + 5 bulbs = 7"
    assert all(step_is_consistent(s) for s in trapped) and answer == "22"
    slipped, _ = write_all(hub, {2: "slip"})
    assert not step_is_consistent(slipped[2])
    order = by_id("r001")  # 15% discount: the trap pays 15% instead of 85%
    trapped, _ = write_all(order, {1: "trap"})
    assert trapped[1] == "Apply 15% discount: 206 x 0.15 = 30.90"


def test_same_prompt_twice_is_a_new_sample_and_runs_are_reproducible() -> None:
    p = by_id("r076")
    a, b = make_reasoner(), make_reasoner()
    first = [a.solve(p).text for _ in range(6)]
    again = [b.solve(p).text for _ in range(6)]
    assert first == again  # deterministic across runs
    assert len(set(first)) > 1  # but each call is a fresh sample


def test_thinking_tokens_are_counted_inside_think_tags() -> None:
    attempt = make_reasoner().solve(by_id("r076"), "think", "high")
    assert attempt.text.startswith("<think>")
    assert 0 < attempt.thinking_tokens < attempt.completion_tokens
    steps, answer, _ = parse_solution(attempt.text)
    assert len(steps) == 5 and answer == attempt.answer


def test_simulator_only_continues_its_own_solutions() -> None:
    p = by_id("r076")
    reasoner = make_reasoner()
    with pytest.raises(ValueError):
        reasoner.next_step(p, ("I made this step up",))
    provider = FakeProvider(responder=SimulatedReasoner())
    with pytest.raises(ValueError):
        provider.complete([Message("system", prompts.SOLVE_COT), Message("user", "Problem: what is 2 + 2?")])
    with pytest.raises(ValueError):
        provider.complete(
            [Message("system", "Be nice."), Message("user", prompts.problem_message(p.question))]
        )


def test_step_by_step_generation_matches_the_chain_bookkeeping() -> None:
    p = by_id("r001")
    reasoner = make_reasoner()
    steps: tuple[str, ...] = ()
    while True:
        step, answer = reasoner.next_step(p, steps)
        if step is None:
            break
        steps = (*steps, step)
    assert len(steps) == p.difficulty
    assert simulator_of(reasoner).chain_of(p, steps).phase == p.difficulty
    assert answer is not None


def test_higher_skill_means_fewer_errors() -> None:
    problems = [p for p in load_problems() if p.is_numeric]
    weak = make_reasoner(SimConfig())
    strong = make_reasoner(
        SimConfig(skill={t: 0.99 for t in SimConfig().skill}, trap={t: 0.0 for t in SimConfig().trap})
    )
    weak_hits = sum(check_answer(p, weak.solve(p).answer) for p in problems)
    strong_hits = sum(check_answer(p, strong.solve(p).answer) for p in problems)
    assert strong_hits > weak_hits
