"""Every experiment behind the numbers quoted in the lessons. The CLI prints them; the tests assert them."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from .problems import check_answer, load_problems, numeric_problems, target_problems
from .reasoner import Attempt, make_reasoner, simulator_of
from .scaling import (
    VoteStats,
    best_of_n,
    majority_vote,
    mean,
    prm_feedback,
    recheck_feedback,
    revise_loop,
    sample,
    self_feedback,
    vote_stats,
)
from .search import prm_beam_search, sample_until_verified, tot_bfs, tot_dfs
from .training import RLStep, StarRound, grpo_toy, split_train_test, star
from .verifiers import SimulatedORM, SimulatedPRM, first_bad_step

SOLVE_MODES = (
    ("direct", "direct", "medium"),
    ("cot", "cot", "medium"),
    ("think-low", "think", "low"),
    ("think-medium", "think", "medium"),
    ("think-high", "think", "high"),
)
SCALE_NS = (1, 3, 5, 9, 15, 45)
HACK_BIAS = 5.0  # length bias of the "hacked" ORM


# --------------------------------------------------------------------------- #
# Lesson 1: direct answer vs chain of thought vs thinking                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SolveRow:
    label: str
    accuracy: float
    by_difficulty: dict[int, float]
    tokens: float  # generated tokens per answer (thinking included)
    thinking: float  # of which inside <think>


@cache
def run_solve() -> tuple[SolveRow, ...]:
    problems = load_problems()
    rows = []
    for label, mode, effort in SOLVE_MODES:
        reasoner = make_reasoner()
        attempts = [reasoner.solve(p, mode, effort) for p in problems]
        hits = [check_answer(p, a.answer) for p, a in zip(problems, attempts, strict=True)]
        by_difficulty = {}
        for d in sorted({p.difficulty for p in problems}):
            sel = [h for p, h in zip(problems, hits, strict=True) if p.difficulty == d]
            by_difficulty[d] = sum(sel) / len(sel)
        rows.append(
            SolveRow(
                label,
                sum(hits) / len(hits),
                by_difficulty,
                mean([a.completion_tokens for a in attempts]),
                mean([a.thinking_tokens for a in attempts]),
            )
        )
    return tuple(rows)


# --------------------------------------------------------------------------- #
# Lesson 2: parallel sampling                                                 #
# --------------------------------------------------------------------------- #


@cache
def numeric_samples(n: int = 45) -> tuple[dict[str, list[Attempt]], dict[str, bool]]:
    """45 CoT samples per numeric problem, plus whether each sample fell into the problem's trap."""
    problems = numeric_problems()
    reasoner = make_reasoner()
    samples = sample(reasoner, problems, n)
    sim = simulator_of(reasoner)
    trapped = {a.text: sim.chain_of(p, a.steps).trapped for p in problems for a in samples[p.id]}
    return samples, trapped


@dataclass(frozen=True)
class ScaleRow:
    n: int
    majority: float
    best_of_n_orm: float
    pass_at_n: float
    tokens: float  # generated tokens per problem


@dataclass(frozen=True)
class ScaleResult:
    rows: tuple[ScaleRow, ...]
    stats: VoteStats  # at the largest N


@cache
def run_scale() -> ScaleResult:
    problems = numeric_problems()
    samples, trapped = numeric_samples()
    orm = SimulatedORM()
    rows = []
    for n in SCALE_NS:
        maj = orm_acc = hit = 0.0
        tokens = 0
        for p in problems:
            subset = samples[p.id][:n]
            maj += check_answer(p, majority_vote([a.answer for a in subset]))
            orm_acc += check_answer(p, best_of_n(subset, [orm.score(p, a) for a in subset]).answer)
            hit += any(check_answer(p, a.answer) for a in subset)
            tokens += sum(a.completion_tokens for a in subset)
        k = len(problems)
        rows.append(ScaleRow(n, maj / k, orm_acc / k, hit / k, tokens / k))
    stats = vote_stats(problems, samples, SCALE_NS[-1], lambda p, a: trapped[a.text])
    return ScaleResult(tuple(rows), stats)


# --------------------------------------------------------------------------- #
# Lesson 2: sequential revision                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReviseRow:
    feedback: str
    accuracy_by_round: tuple[float, ...]
    tokens: float  # generated tokens per problem, all rounds
    calls: float
    broken: int  # right after round 0, wrong at the end


@cache
def run_revise(rounds: int = 3) -> tuple[ReviseRow, ...]:
    problems = numeric_problems()
    rows = []
    for name in ("self-critique", "arithmetic re-check", "process reward model"):
        reasoner = make_reasoner()
        feedback = {
            "self-critique": self_feedback(reasoner),
            "arithmetic re-check": recheck_feedback,
            "process reward model": prm_feedback(SimulatedPRM()),
        }[name]
        correct = [0] * (rounds + 1)
        broken = 0
        for p in problems:
            history = revise_loop(reasoner, p, reasoner.solve(p, "cot"), feedback, rounds)
            for i, attempt in enumerate(history):
                correct[i] += check_answer(p, attempt.answer)
            broken += check_answer(p, history[0].answer) and not check_answer(p, history[-1].answer)
        k = len(problems)
        rows.append(
            ReviseRow(
                name,
                tuple(c / k for c in correct),
                reasoner.usage.completion_tokens / k,
                reasoner.usage.calls / k,
                broken,
            )
        )
    return tuple(rows)


# --------------------------------------------------------------------------- #
# Lesson 3: verifiers and search                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MethodRow:
    method: str
    accuracy: float
    tokens: float  # generated tokens per problem
    prompt_tokens: float
    calls: float


@dataclass(frozen=True)
class VerifierErrors:
    samples: int
    wrong: int
    wrong_trap: int  # wrong because of a trap (tidy arithmetic)
    recheck_caught: int  # wrong samples the arithmetic re-check flags
    recheck_rejects_right: int  # right answers whose written steps do not add up (unfaithful)
    orm_accepts_wrong: int  # wrong samples with ORM score >= 0.5
    orm_rejects_right: int
    prm_accepts_wrong: int  # wrong samples with min step score >= 0.5
    prm_rejects_right: int


@dataclass(frozen=True)
class HackRow:
    n: int
    honest: float
    hacked: float
    honest_tokens: float  # length of the chosen answer
    hacked_tokens: float


@dataclass(frozen=True)
class SearchResultTable:
    methods: tuple[MethodRow, ...]
    errors: VerifierErrors
    hack: tuple[HackRow, ...]


@cache
def run_search(n: int = 9, beam_width: int = 3, expand: int = 3) -> SearchResultTable:
    problems = numeric_problems()
    samples, trapped = numeric_samples()
    k = len(problems)
    orm, prm = SimulatedORM(), SimulatedPRM()
    probe = make_reasoner()  # one CoT call per problem, only to measure the prompt size
    for p in problems:
        probe.solve(p, "cot")
    prompt_one = probe.usage.prompt_tokens / k

    def row(name: str, picks: list[str | None], m: int) -> MethodRow:
        acc = sum(check_answer(p, a) for p, a in zip(problems, picks, strict=True)) / k
        gen = sum(sum(a.completion_tokens for a in samples[p.id][:m]) for p in problems) / k
        return MethodRow(name, acc, gen, prompt_one * m, float(m))

    def first(m: int) -> dict[str, list[Attempt]]:
        return {p.id: samples[p.id][:m] for p in problems}

    top = first(n)
    filtered = []
    for p in problems:
        kept = [a for a in top[p.id] if first_bad_step(a.steps) is None] or top[p.id]
        filtered.append(majority_vote([a.answer for a in kept]))
    rows = [
        row("CoT, 1 sample", [samples[p.id][0].answer for p in problems], 1),
        row(f"majority vote, N={n}", [majority_vote([a.answer for a in top[p.id]]) for p in problems], n),
        row(f"re-check filter + vote, N={n}", filtered, n),
        row(
            f"best-of-{n}, ORM",
            [best_of_n(top[p.id], [orm.score(p, a) for a in top[p.id]]).answer for p in problems],
            n,
        ),
        row(
            f"best-of-{n}, PRM",
            [best_of_n(top[p.id], [prm.score(p, a.steps) for a in top[p.id]]).answer for p in problems],
            n,
        ),
    ]
    beam_reasoner = make_reasoner()
    results = [prm_beam_search(beam_reasoner, prm, p, beam_width=beam_width, expand=expand) for p in problems]
    rows.append(
        MethodRow(
            f"PRM beam search, {beam_width}x{expand}",
            sum(r.solved for r in results) / k,
            sum(r.usage.completion_tokens for r in results) / k,
            sum(r.usage.prompt_tokens for r in results) / k,
            sum(r.usage.calls for r in results) / k,
        )
    )
    rows.append(
        row(
            f"pass@{n} (needs the answer key)",
            [next((a.answer for a in top[p.id] if check_answer(p, a.answer)), None) for p in problems],
            n,
        )
    )

    all_samples = [(p, a) for p in problems for a in samples[p.id]]
    wrong = [(p, a) for p, a in all_samples if not check_answer(p, a.answer)]
    right = [(p, a) for p, a in all_samples if check_answer(p, a.answer)]
    errors = VerifierErrors(
        samples=len(all_samples),
        wrong=len(wrong),
        wrong_trap=sum(trapped[a.text] for _, a in wrong),
        recheck_caught=sum(first_bad_step(a.steps) is not None for _, a in wrong),
        recheck_rejects_right=sum(first_bad_step(a.steps) is not None for _, a in right),
        orm_accepts_wrong=sum(orm.score(p, a) >= 0.5 for p, a in wrong),
        orm_rejects_right=sum(orm.score(p, a) < 0.5 for p, a in right),
        prm_accepts_wrong=sum(prm.score(p, a.steps) >= 0.5 for p, a in wrong),
        prm_rejects_right=sum(prm.score(p, a.steps) < 0.5 for p, a in right),
    )

    hacked = SimulatedORM(length_bias=HACK_BIAS)
    hack_rows = []
    for m in SCALE_NS:
        honest_pick = [
            best_of_n(samples[p.id][:m], [orm.score(p, a) for a in samples[p.id][:m]]) for p in problems
        ]
        hacked_pick = [
            best_of_n(samples[p.id][:m], [hacked.score(p, a) for a in samples[p.id][:m]]) for p in problems
        ]
        hack_rows.append(
            HackRow(
                m,
                sum(check_answer(p, a.answer) for p, a in zip(problems, honest_pick, strict=True)) / k,
                sum(check_answer(p, a.answer) for p, a in zip(problems, hacked_pick, strict=True)) / k,
                mean([a.completion_tokens for a in honest_pick]),
                mean([a.completion_tokens for a in hacked_pick]),
            )
        )
    return SearchResultTable(tuple(rows), errors, tuple(hack_rows))


# --------------------------------------------------------------------------- #
# Lesson 3: Tree of Thoughts on the target puzzles                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TotRow:
    method: str
    solved: int
    tokens: float
    prompt_tokens: float
    calls: float
    rejected: int


@cache
def run_tot() -> tuple[TotRow, ...]:
    problems = target_problems()
    methods = (
        ("CoT, 1 sample", lambda r, p: sample_until_verified(r, p, 1)),
        ("sample until verified, max 5", lambda r, p: sample_until_verified(r, p, 5)),
        ("sample until verified, max 15", lambda r, p: sample_until_verified(r, p, 15)),
        ("ToT BFS, breadth 1", lambda r, p: tot_bfs(r, p, breadth=1)),
        ("ToT BFS, breadth 3", lambda r, p: tot_bfs(r, p, breadth=3)),
        ("ToT BFS, breadth 5", lambda r, p: tot_bfs(r, p, breadth=5)),
        ("ToT DFS with pruning", lambda r, p: tot_dfs(r, p)),
    )
    rows = []
    k = len(problems)
    for name, fn in methods:
        reasoner = make_reasoner()
        results = [fn(reasoner, p) for p in problems]
        rows.append(
            TotRow(
                name,
                sum(r.solved for r in results),
                sum(r.usage.completion_tokens for r in results) / k,
                sum(r.usage.prompt_tokens for r in results) / k,
                sum(r.usage.calls for r in results) / k,
                sum(r.rejected for r in results),
            )
        )
    return tuple(rows)


# --------------------------------------------------------------------------- #
# Lesson 4: STaR and RL                                                       #
# --------------------------------------------------------------------------- #


@cache
def run_star(rationalize: bool = True, strict: bool = False, rounds: int = 4) -> tuple[StarRound, ...]:
    train, test = split_train_test(load_problems())
    return tuple(star(train, test, rounds=rounds, rationalize=rationalize, strict=strict))


@cache
def run_rl(steps: int = 8, lr: float = 40.0, group: int = 8) -> tuple[RLStep, ...]:
    train, test = split_train_test(load_problems())
    return tuple(grpo_toy(train, test, steps=steps, lr=lr, group=group))
