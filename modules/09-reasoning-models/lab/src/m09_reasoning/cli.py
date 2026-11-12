"""Command line: ``uv run lab <command>``. Every command is offline and deterministic by default."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import experiments as ex
from .problems import check_answer, load_problems
from .reasoner import make_reasoner, simulator_of
from .search import prm_beam_search, tot_bfs, tot_dfs
from .verifiers import SimulatedPRM

NOTE = "(simulated reasoner: numbers show how the algorithms behave, not how any real model scores)"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def num(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{x:,.{digits}f}"


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[str(c) for c in headers]] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = []
    for n, r in enumerate(cells):
        out = [r[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(r[1:], widths[1:], strict=True)]
        lines.append("  ".join(out).rstrip())
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _problem(pid: str):
    for p in load_problems():
        if p.id == pid:
            return p
    raise SystemExit(f"unknown problem id {pid!r} (ids go from r001 to r160)")


# --------------------------------------------------------------------------- #
# Commands                                                                    #
# --------------------------------------------------------------------------- #


def cmd_solve(args: argparse.Namespace) -> None:
    if args.id:
        p = _problem(args.id)
        reasoner = make_reasoner(provider=args.provider)
        print(f"{p.id} · {p.type} · difficulty {p.difficulty}")
        print(f"{p.question}\nReference answer: {p.answer}\n")
        for label, mode, effort in ex.SOLVE_MODES:
            if args.mode and args.mode != label:
                continue
            attempt = reasoner.solve(p, mode, effort)
            verdict = "correct" if check_answer(p, attempt.answer) else "wrong"
            print(
                f"=== {label}: {attempt.completion_tokens} tokens "
                f"({attempt.thinking_tokens} thinking), {verdict}"
            )
            print(attempt.text)
            if args.provider in (None, "fake") and attempt.steps:
                kinds = simulator_of(reasoner).chain_of(p, attempt.steps).kinds
                print(f"[behind the curtain: step kinds {', '.join(kinds)}]")
            print()
        return
    if args.provider not in (None, "fake"):
        reasoner = make_reasoner(provider=args.provider)
        problems = load_problems()[: args.limit]
        mode = args.mode or "cot"
        hits = sum(
            check_answer(p, reasoner.solve(p, mode.split("-")[0], (mode.split("-") + ["medium"])[1]).answer)
            for p in problems
        )
        print(
            f"{mode} with {args.provider}: {hits}/{len(problems)} correct, "
            f"{reasoner.usage.completion_tokens} generated tokens"
        )
        return
    rows = ex.run_solve()
    print(f"One sample per problem, all 160 problems {NOTE}\n")
    print(
        table(
            [
                "mode",
                "accuracy",
                "diff 2",
                "diff 3",
                "diff 4",
                "diff 5",
                "tokens/answer",
                "of which thinking",
            ],
            [
                [
                    r.label,
                    pct(r.accuracy),
                    *(pct(r.by_difficulty[d]) for d in (2, 3, 4, 5)),
                    num(r.tokens),
                    num(r.thinking),
                ]
                for r in rows
            ],
        )
    )


def cmd_scale(_: argparse.Namespace) -> None:
    result = ex.run_scale()
    print(f"Parallel sampling on the 110 numeric problems {NOTE}\n")
    print(
        table(
            ["N", "majority vote", "best-of-N (ORM)", "pass@N (oracle)", "tokens/problem"],
            [
                [r.n, pct(r.majority), pct(r.best_of_n_orm), pct(r.pass_at_n), num(r.tokens)]
                for r in result.rows
            ],
        )
    )
    s = result.stats
    print(
        f"\nAt N={ex.SCALE_NS[-1]}: the most frequent answer is wrong on {s.plurality_wrong} of {s.problems} "
        f"problems; {s.wrong_from_trap} of them are trap answers. Problems never solved: {s.never_right}."
    )


def cmd_revise(_: argparse.Namespace) -> None:
    rows = ex.run_revise()
    print(f"Sequential revision on the 110 numeric problems, 3 rounds {NOTE}\n")
    print(
        table(
            [
                "feedback",
                "round 0",
                "round 1",
                "round 2",
                "round 3",
                "tokens/problem",
                "calls/problem",
                "right->wrong",
            ],
            [
                [r.feedback, *(pct(a) for a in r.accuracy_by_round), num(r.tokens), num(r.calls, 2), r.broken]
                for r in rows
            ],
        )
    )


def cmd_search(args: argparse.Namespace) -> None:
    if args.id:
        p = _problem(args.id)
        if not p.is_numeric:
            raise SystemExit("search --id works on numeric problems; use 'lab tot --id' for target puzzles")
        result = prm_beam_search(make_reasoner(), SimulatedPRM(), p)
        print(f"{p.id}: {p.question}\n")
        print("\n".join(result.trace))
        print(
            f"\nAnswer {result.answer} ({'correct' if result.solved else 'wrong'}), "
            f"{result.usage.calls} calls, {result.usage.completion_tokens} generated tokens, "
            f"{result.usage.prompt_tokens} prompt tokens"
        )
        return
    result = ex.run_search()
    print(f"Verifiers and search on the 110 numeric problems {NOTE}\n")
    print(
        table(
            ["method", "accuracy", "generated tok", "prompt tok", "calls"],
            [
                [r.method, pct(r.accuracy), num(r.tokens), num(r.prompt_tokens), num(r.calls)]
                for r in result.methods
            ],
        )
    )
    e = result.errors
    print(
        f"\nVerifier errors over all {e.samples:,} samples ({e.wrong:,} wrong, {e.wrong_trap} of them traps):"
    )
    print(
        table(
            ["verifier", "wrong it lets through", "right it rejects"],
            [
                ["arithmetic re-check", e.wrong - e.recheck_caught, e.recheck_rejects_right],
                ["ORM (score >= 0.5)", e.orm_accepts_wrong, e.orm_rejects_right],
                ["PRM (min step >= 0.5)", e.prm_accepts_wrong, e.prm_rejects_right],
            ],
        )
    )
    print(
        f"\nReward hacking: best-of-N with an honest ORM vs one that also likes long answers "
        f"(length bias {ex.HACK_BIAS})"
    )
    print(
        table(
            ["N", "honest ORM", "length-biased ORM", "tokens of pick (honest)", "tokens of pick (biased)"],
            [
                [r.n, pct(r.honest), pct(r.hacked), num(r.honest_tokens), num(r.hacked_tokens)]
                for r in result.hack
            ],
        )
    )


def cmd_tot(args: argparse.Namespace) -> None:
    if args.id:
        p = _problem(args.id)
        if p.type != "target":
            raise SystemExit("tot --id needs a target puzzle (r111 to r160)")
        print(f"{p.id}: {p.question}\n")
        for name, fn in (
            ("BFS, breadth 5", lambda r: tot_bfs(r, p, breadth=5)),
            ("DFS", lambda r: tot_dfs(r, p)),
        ):
            result = fn(make_reasoner())
            print(f"=== {name}")
            print("\n".join(result.trace) or "(no backtracking)")
            print(
                f"found: {result.answer}  ·  {result.usage.calls} calls, "
                f"{result.usage.completion_tokens} generated tokens\n"
            )
        return
    rows = ex.run_tot()
    print(f"The 50 target puzzles, every answer checked by the exact verifier {NOTE}\n")
    print(
        table(
            ["method", "solved", "generated tok", "prompt tok", "calls", "verifier rejections"],
            [
                [r.method, f"{r.solved}/50", num(r.tokens), num(r.prompt_tokens), num(r.calls), r.rejected]
                for r in rows
            ],
        )
    )


def cmd_star(args: argparse.Namespace) -> None:
    rows = ex.run_star(rationalize=not args.no_rationalize, strict=args.strict, rounds=args.rounds)
    how = "with" if not args.no_rationalize else "without"
    print(
        f"STaR on 80 training problems, accuracy on 80 held-out problems, {how} rationalisation"
        f"{', strict filter' if args.strict else ''} {NOTE}\n"
    )
    families = list(rows[0].by_family)
    print(
        table(
            [
                "round",
                "test accuracy",
                *families,
                "kept (solved)",
                "kept (hint)",
                "unfaithful kept",
                "tokens spent",
            ],
            [
                [
                    r.round,
                    pct(r.test_accuracy),
                    *(pct(r.by_family[f]) for f in families),
                    num(r.kept_solved, 0),
                    num(r.kept_hint, 0),
                    num(r.unfaithful, 0),
                    num(r.generated_tokens, 0),
                ]
                for r in rows
            ],
        )
    )


def cmd_rl(args: argparse.Namespace) -> None:
    rows = ex.run_rl(steps=args.steps, lr=args.lr, group=args.group)
    print(
        f"GRPO-style toy: groups of {args.group} samples per training problem, "
        "reward 1 if the answer is right "
        f"{NOTE}\n"
    )
    print(
        table(
            [
                "step",
                "train reward",
                "test accuracy",
                "tokens/answer",
                "P(check step)",
                "P(re-read)",
                "groups with no signal",
            ],
            [
                [
                    r.step,
                    num(r.train_reward, 3),
                    pct(r.test_accuracy),
                    num(r.tokens),
                    num(r.check_prob, 2),
                    num(r.reread_prob, 2),
                    num(r.zero_signal_groups, 0),
                ]
                for r in rows
            ],
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lab", description="Module 09 lab: reasoning models (offline).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("solve", help="direct answer vs chain of thought vs thinking (lesson 1)")
    p.add_argument("--id", help="show the full output for one problem, e.g. r076")
    p.add_argument("--mode", choices=[label for label, _, _ in ex.SOLVE_MODES], help="only this mode")
    p.add_argument(
        "--provider",
        choices=["fake", "ollama", "openai"],
        default=None,
        help="default: fake (the simulator); real models are optional and untested here",
    )
    p.add_argument("--limit", type=int, default=160, help="problems to run with a real provider")
    p.set_defaults(func=cmd_solve)

    sub.add_parser("scale", help="self-consistency and best-of-N curves (lesson 2)").set_defaults(
        func=cmd_scale
    )
    sub.add_parser("revise", help="sequential revision with feedback (lesson 2)").set_defaults(
        func=cmd_revise
    )

    p = sub.add_parser("search", help="verifiers, ORM vs PRM, beam search, reward hacking (lesson 3)")
    p.add_argument("--id", help="trace PRM beam search on one numeric problem, e.g. r076")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("tot", help="Tree of Thoughts on the target puzzles (lesson 3)")
    p.add_argument("--id", help="trace the search on one puzzle, e.g. r111")
    p.set_defaults(func=cmd_tot)

    p = sub.add_parser("star", help="Self-Taught Reasoner loop (lesson 4)")
    p.add_argument(
        "--no-rationalize", action="store_true", help="skip rationalisation with the answer as hint"
    )
    p.add_argument(
        "--strict", action="store_true", help="also require every step to pass the arithmetic re-check"
    )
    p.add_argument("--rounds", type=int, default=4)
    p.set_defaults(func=cmd_star)

    p = sub.add_parser("rl", help="RL with a verifiable reward, GRPO-style toy (lesson 4)")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=40.0)
    p.add_argument("--group", type=int, default=8, help="samples per training problem")
    p.set_defaults(func=cmd_rl)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
