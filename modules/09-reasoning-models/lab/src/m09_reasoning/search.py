"""Search at inference time: PRM beam search, Tree of Thoughts (BFS/DFS), sampling until verified."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .problems import Problem, check_answer, verify_expression
from .puzzle import Item, exact, strip_outer
from .reasoner import Reasoner, Usage
from .verifiers import SimulatedPRM

# How a "sure / likely / impossible" judgement turns into a number for ranking states.
VALUE_SCORE = {"sure": 20.0, "likely": 1.0, "impossible": 0.001}


@dataclass
class SearchResult:
    answer: str | None
    solved: bool
    usage: Usage
    rejected: int = 0  # candidates the exact verifier threw away
    trace: list[str] = field(default_factory=list)


def prm_beam_search(
    reasoner: Reasoner,
    prm: SimulatedPRM,
    problem: Problem,
    *,
    beam_width: int = 3,
    expand: int = 3,
    max_depth: int = 10,
) -> SearchResult:
    """Step-level beam search: extend each beam ``expand`` times, keep the ``beam_width`` best by PRM."""
    start = reasoner.usage.snapshot()
    beams: list[tuple[str, ...]] = [()]
    finished: list[tuple[float, tuple[str, ...], str | None]] = []
    trace: list[str] = []
    for depth in range(max_depth):
        candidates: dict[tuple[str, ...], float] = {}
        for steps in beams:
            for _ in range(expand):
                step, answer = reasoner.next_step(problem, steps)
                if step is None:  # this beam is complete
                    finished.append((prm.score(problem, steps), steps, answer))
                    break
                new = (*steps, step)
                if new not in candidates:  # the same continuation sampled twice is one candidate
                    candidates[new] = prm.score(problem, new)
        if not candidates:
            break
        ranked = sorted(candidates, key=lambda s: -candidates[s])
        beams = ranked[:beam_width]
        trace.append(f"depth {depth + 1}: {len(candidates)} distinct candidates, keep {len(beams)}")
        for steps in ranked:
            mark = "keep" if steps in beams else "drop"
            trace.append(f"  {mark} {candidates[steps]:.2f}  {steps[-1]}")
    score, _, answer = max(finished, key=lambda f: f[0])
    return SearchResult(answer, check_answer(problem, answer), reasoner.usage.minus(start), trace=trace)


def _leaf_expression(state: list[Item]) -> str:
    return strip_outer(state[0][1])


def _start(problem: Problem) -> list[Item]:
    return [(Fraction(n), str(n)) for n in problem.numbers]


def _left(state: list[Item]) -> str:
    return "[" + " ".join(exact(v) for v, _ in state) + "]"


def tot_bfs(reasoner: Reasoner, problem: Problem, *, breadth: int = 5, k: int = 5) -> SearchResult:
    """Tree of Thoughts, breadth-first: propose k moves per state, rate them, keep the ``breadth`` best."""
    assert problem.target is not None
    start = reasoner.usage.snapshot()
    frontier = [_start(problem)]
    trace: list[str] = []
    rejected = 0
    while len(frontier[0]) > 2:
        scored: list[tuple[float, str, list[Item]]] = []
        for state in frontier:
            for child in reasoner.propose(problem, state, k):
                word = reasoner.value(problem, child)
                scored.append((VALUE_SCORE[word], word, child))
        scored.sort(key=lambda t: -t[0])
        frontier = [child for _, _, child in scored[:breadth]]
        kept = ", ".join(f"{_left(c)} {w}" for _, w, c in scored[:breadth])
        trace.append(f"{len(scored)} proposals rated, kept {len(frontier)}: {kept}")
        if not frontier:
            return SearchResult(None, False, reasoner.usage.minus(start), trace=trace)
    for state in frontier:  # last move: no rating needed, the exact verifier decides
        for leaf in reasoner.propose(problem, state, k):
            expr = _leaf_expression(leaf)
            if verify_expression(problem.numbers, problem.target, expr):
                return SearchResult(expr, True, reasoner.usage.minus(start), rejected, trace)
            rejected += leaf[0][0] == problem.target  # claimed the target, but the verifier disagrees
    return SearchResult(None, False, reasoner.usage.minus(start), rejected, trace)


def tot_dfs(reasoner: Reasoner, problem: Problem, *, k: int = 5, max_calls: int = 60) -> SearchResult:
    """Tree of Thoughts, depth-first: follow the best-rated move, prune 'impossible', backtrack on failure."""
    assert problem.target is not None
    target = problem.target
    start = reasoner.usage.snapshot()
    rejected = [0]
    trace: list[str] = []

    def visit(state: list[Item], depth: int) -> str | None:
        if reasoner.usage.calls - start.calls >= max_calls:
            return None
        children = reasoner.propose(problem, state, k)
        if len(state) == 2:
            for leaf in children:
                expr = _leaf_expression(leaf)
                if verify_expression(problem.numbers, target, expr):
                    return expr
                rejected[0] += leaf[0][0] == target
            trace.append(f"{'  ' * depth}no last move reaches the target: backtrack")
            return None
        rated = [(reasoner.value(problem, c), i, c) for i, c in enumerate(children)]
        for word, _, child in sorted(rated, key=lambda t: (-VALUE_SCORE[t[0]], t[1])):
            if word == "impossible":
                trace.append(f"{'  ' * depth}prune {_left(child)} (rated impossible)")
                continue
            trace.append(f"{'  ' * depth}try {_left(child)} (rated {word})")
            found = visit(child, depth + 1)
            if found is not None:
                return found
        return None

    expr = visit(_start(problem), 0)
    return SearchResult(expr, expr is not None, reasoner.usage.minus(start), rejected[0], trace)


def sample_until_verified(reasoner: Reasoner, problem: Problem, max_samples: int) -> SearchResult:
    """Draw CoT samples until one passes the exact verifier (only possible when a verifier exists)."""
    start = reasoner.usage.snapshot()
    rejected = 0
    for _ in range(max_samples):
        attempt = reasoner.solve(problem, "cot")
        if check_answer(problem, attempt.answer):
            return SearchResult(attempt.answer, True, reasoner.usage.minus(start), rejected)
        rejected += 1
    return SearchResult(None, False, reasoner.usage.minus(start), rejected)
