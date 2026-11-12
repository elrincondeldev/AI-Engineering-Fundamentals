"""Voting, best-of-N, revision, search and the training loops."""

from __future__ import annotations

import math

from m09_reasoning.problems import load_problems, verify_expression
from m09_reasoning.reasoner import Attempt, make_reasoner
from m09_reasoning.scaling import answer_key, best_of_n, majority_vote, recheck_feedback, revise_loop
from m09_reasoning.search import prm_beam_search, sample_until_verified, tot_bfs, tot_dfs
from m09_reasoning.simulator import SimConfig
from m09_reasoning.training import finetune, group_advantages, split_train_test
from m09_reasoning.verifiers import SimulatedPRM


def by_id(pid: str):
    return next(p for p in load_problems() if p.id == pid)


def test_majority_vote_normalises_and_breaks_ties_by_order() -> None:
    assert answer_key("175.10") == answer_key("175.1")
    assert majority_vote(["175.1", "90", "175.10"]) == "175.1"
    assert majority_vote(["52", "22"]) == "52"
    assert majority_vote([None, None]) is None


def test_best_of_n_takes_the_highest_score_first_on_ties() -> None:
    a, b, c = (Attempt(t, (), t, 1) for t in "abc")
    assert best_of_n([a, b, c], [0.2, 0.9, 0.9]) is b


def test_group_advantages() -> None:
    adv = group_advantages([1, 0, 0, 1])
    assert math.isclose(sum(adv), 0.0, abs_tol=1e-12) and adv[0] > 0 > adv[1]
    assert group_advantages([1, 1, 1]) == [0.0, 0.0, 0.0]  # no signal when every sample agrees


def test_revision_stops_when_the_verifier_is_happy() -> None:
    p = by_id("r002")
    reasoner = make_reasoner()
    right = Attempt("x", p.steps, p.answer, 1)
    history = revise_loop(reasoner, p, right, recheck_feedback, rounds=3)
    assert len(history) == 4 and all(h is right for h in history)
    assert reasoner.usage.calls == 0


def test_tree_of_thoughts_answers_always_pass_the_verifier() -> None:
    for pid in ("r111", "r112", "r126"):
        p = by_id(pid)
        for result in (tot_bfs(make_reasoner(), p), tot_dfs(make_reasoner(), p)):
            if result.solved:
                assert verify_expression(p.numbers, p.target, result.answer)
    assert tot_dfs(make_reasoner(), by_id("r112")).solved


def test_sampling_until_verified_stops_at_the_first_success() -> None:
    p = by_id("r112")
    result = sample_until_verified(make_reasoner(), p, 15)
    assert result.solved and result.usage.calls == result.rejected + 1


def test_beam_search_returns_a_complete_solution() -> None:
    p = by_id("r076")
    result = prm_beam_search(make_reasoner(), SimulatedPRM(), p)
    assert result.answer is not None and result.usage.calls > p.difficulty


def test_split_and_finetune() -> None:
    train, test = split_train_test(load_problems())
    assert len(train) == len(test) == 80 and not {p.id for p in train} & {p.id for p in test}
    base = SimConfig()
    assert finetune(base, [], train, "same").skill == base.skill
