"""Command line for the lab: ``uv run lab --help``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import Result
from .evaluate import Graded, grade, load_tasks, summarize
from .judge import agreement, run_probes
from .multi import MultiAgentSystem
from .runner import AGENTS, live_provider, replay, run_agent, run_eval, save_run
from .sanitize import sanitize
from .search import Guards
from .tools import WebTools
from .web import load_snapshot

ABLATIONS = [
    ("all guards on", Guards()),
    ("no dedup", Guards(dedup=False)),
    ("no credibility", Guards(credibility=False)),
    ("no recency", Guards(recency=False)),
    ("no sanitize", Guards(sanitize=False)),
    ("all guards off", Guards.none()),
]


def guards_from(args: argparse.Namespace) -> Guards:
    if args.naive:
        return Guards.none()
    return Guards(not args.no_dedup, not args.no_credibility, not args.no_recency, not args.no_sanitize)


def print_trace(r: Result) -> None:
    for i, s in enumerate(r.steps, 1):
        first = s.observation.splitlines()[0] if s.observation else ""
        if s.tool == "search_web":
            n = s.observation.count("    url: ")
            first = f"{n} results, top: " + (s.observation.split("url: ")[1].split(" |")[0] if n else "-")
        elif s.tool == "open_page" and "\n" in s.observation:
            first = s.observation.splitlines()[2]
        print(f"Step {i:>2}  {s.tool}[{s.arg}]")
        print(f"         thought: {s.thought}")
        print(f"         -> {first}")
    for note in r.notes:
        print(f"note: {note.splitlines()[0] if note.startswith('Plan:') else note}")


def print_bill(r: Result) -> None:
    u = r.usage
    print(
        f"Cost: {u.calls} model calls · {r.tool_calls} tool calls · {u.tokens:,} tokens "
        f"(prompt {u.prompt_tokens:,} + completion {u.completion_tokens:,})"
    )


def print_summary(agent: str, harness: str, graded: list[Graded]) -> None:
    s = summarize(graded)
    print(f"Agent {agent} · {harness} · {s.n} tasks")
    print(f"{'type':<14} success")
    for t, (ok, n) in s.by_type.items():
        print(f"{t:<14} {ok}/{n}")
    ok = round(s.success * s.n)
    print(f"Success (correct and safe): {ok}/{s.n} = {s.success:.3f}")
    print(f"Exact match: {s.exact:.3f} · lenient match: {s.correct:.3f}")
    print(
        f"Unanswerable answered 'unknown': {s.unknown_correct[0]}/{s.unknown_correct[1]} · "
        f"false 'unknown' on answerable: {s.false_unknown}"
    )
    print(
        f"Citation precision (gold URL and supports the claim): {s.citation_precision:.3f} "
        f"over {s.citations} citations"
    )
    print(
        f"Avg tool calls: {s.avg_steps:.2f} · avg model calls: {s.avg_model_calls:.2f} · "
        f"avg tokens: {s.avg_tokens:,.0f}"
    )
    print(
        f"Tool errors: {s.tool_errors} · redundant calls: {s.redundant_calls} · "
        f"must_not_contain violations: {s.violations} · uncited domains: {s.foreign_domains}"
    )
    print(f"Failed: {', '.join(s.failed_ids) or 'none'}")


def cmd_search(args: argparse.Namespace) -> None:
    g = guards_from(args)
    print(f"({g.label()})")
    print(WebTools(g).search_web(args.query, k=args.k))


def cmd_open(args: argparse.Namespace) -> None:
    g = guards_from(args)
    tools = WebTools(g)
    print(tools.open_page(args.url))
    page = load_snapshot().get(args.url)
    if page and page.ok:
        clean = sanitize(page.text)
        if g.sanitize:
            print(f"(sanitiser applied: {clean.removed} sentence(s) removed, {clean.masked} item(s) masked)")
        else:
            print(f"(sanitiser OFF: it would remove {clean.removed} sentence(s) and mask {clean.masked} item(s))")


def options_from(args: argparse.Namespace) -> dict:
    return {
        "max_steps": args.max_steps,
        "branching": args.branching,
        "beam": args.beam,
        "critic": not args.no_critic,
        "structured_handoff": not args.free_text_handoff,
    }


def run_and_show(args: argparse.Namespace, question: str) -> Result:
    g = guards_from(args)
    print(f"Agent: {args.agent} · {g.label()}")
    if args.agent == "multi":
        system = MultiAgentSystem(
            WebTools(g), critic=not args.no_critic, structured_handoff=not args.free_text_handoff
        )
        r = system.run(question)
        if args.trace:
            print_wire(system)
    else:
        r = run_agent(args.agent, question, g, **options_from(args))
        if args.trace:
            print_trace(r)
    print(r.answer_text)
    print_bill(r)
    return r


def print_wire(system: MultiAgentSystem) -> None:
    for card in system.directory.cards.values():
        print(f"card: {card.name} · skills {list(card.skills)} · {card.description}")
    for env in system.directory.wire:
        if "method" in env:
            msg = env["params"]["message"]["parts"][0]["content"]
            short = msg if isinstance(msg, str) else f"{len(msg)} claim(s)"
            print(f"-> #{env['id']} {env['method']} to {env['params']['agent']}: {short}")
        else:
            print(f"<- #{env['id']} {env['result']['task']} state={env['result']['state']}")
    for note in system.result.notes:
        print(f"note: {note}")


def cmd_ask(args: argparse.Namespace) -> None:
    run_and_show(args, args.question)


def cmd_task(args: argparse.Namespace) -> None:
    tasks = {t["id"]: t for t in load_tasks()}
    if args.task_id not in tasks:
        raise SystemExit(f"Unknown task {args.task_id!r} (task01 ... task24).")
    t = tasks[args.task_id]
    print(f"{t['id']} ({t['type']}): {t['question']}")
    r = run_and_show(args, t["question"])
    gr = grade(r, t)
    print(
        f"Expected: {t['answer']} · correct: {gr.correct} · exact: {gr.exact} · "
        f"violations: {gr.violations or 'none'} · citations ok: {sum(gr.citations)}/{len(gr.citations)}"
    )


def agent_label(args: argparse.Namespace) -> str:
    extras = {
        "react": [f"max_steps={args.max_steps}"] if args.max_steps != 12 else [],
        "tree": [f"branching={args.branching}"] * (args.branching != 3)
        + [f"beam={args.beam}"] * (args.beam != 2),
        "mcts": [f"branching={args.branching}"] * (args.branching != 3),
        "multi": ["no critic"] * args.no_critic + ["free-text handoff"] * args.free_text_handoff,
    }
    opts = extras.get(args.agent, [])
    return f"{args.agent} ({', '.join(opts)})" if opts else args.agent


def cmd_eval(args: argparse.Namespace) -> None:
    g = guards_from(args)
    graded = run_eval(args.agent, g, **options_from(args))
    print_summary(agent_label(args), g.label(), graded)
    if args.save:
        save_run(Path(args.save), graded, g)
        print(f"Saved {len(graded)} records to {args.save}")


def cmd_compare(args: argparse.Namespace) -> None:
    g = guards_from(args)
    print(f"All agents · {g.label()} · 24 tasks")
    head = ("agent", "success", "exact", "cite", "tools", "calls", "tokens", "viol")
    print("{:<10} {:>7} {:>6} {:>6} {:>6} {:>6} {:>7} {:>5}  failed".format(*head))
    for agent in AGENTS:
        s = summarize(run_eval(agent, g))
        failed = ", ".join(s.failed_ids) or "-"
        print(
            f"{agent:<10} {s.success:>7.3f} {s.exact:>6.3f} {s.citation_precision:>6.3f} {s.avg_steps:>6.2f} "
            f"{s.avg_model_calls:>6.2f} {s.avg_tokens:>7,.0f} {s.violations:>5}  {failed}"
        )


def cmd_ablate(args: argparse.Namespace) -> None:
    print(f"Ablation · agent {args.agent} · 24 tasks")
    print(f"{'harness':<16} {'success':>7} {'viol':>5} {'tools':>6}  failed")
    for name, g in ABLATIONS:
        s = summarize(run_eval(args.agent, g))
        failed = ", ".join(s.failed_ids) or "-"
        print(f"{name:<16} {s.success:>7.3f} {s.violations:>5} {s.avg_steps:>6.2f}  {failed}")


def cmd_judge(args: argparse.Namespace) -> None:
    rows = run_probes(live_provider())
    print(f"{'task':<7} {'human':>5} {'exact':>5} {'lenient':>7} {'judge':>5} {'viol':>4}  candidate")
    yn = {True: "yes", False: "no"}
    for r in rows:
        print(
            f"{r.task_id:<7} {yn[r.human]:>5} {yn[r.exact]:>5} {yn[r.lenient]:>7} {yn[r.judge]:>5} "
            f"{yn[r.violation]:>4}  {r.candidate}"
        )
    n = len(rows)
    for grader in ("exact", "lenient", "judge"):
        print(
            f"{grader:<8} agrees with the human on {agreement(rows, grader)}/{n}; "
            f"with must_not_contain as a veto: {agreement(rows, grader, True)}/{n}"
        )


def cmd_replay(args: argparse.Namespace) -> None:
    path = Path(args.path)
    graded = replay(path)
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0]) if graded else {}
    print(f"Replayed {len(graded)} saved records from {args.path} (no model or tool calls)")
    print_summary(first.get("agent", "?"), first.get("guards", "?"), graded)


def add_agent_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-steps", type=int, default=12, help="react: tool-call budget")
    p.add_argument("--branching", type=int, default=3, help="tree/mcts: results considered per decision")
    p.add_argument("--beam", type=int, default=2, help="tree: partial answers kept per level")
    p.add_argument("--no-critic", action="store_true", help="multi: skip the fact-checker")
    p.add_argument("--free-text-handoff", action="store_true", help="multi: pass free text between agents")


def add_guard_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--naive", action="store_true", help="switch every guard off")
    p.add_argument("--no-dedup", action="store_true", help="keep duplicate and mirrored pages")
    p.add_argument("--no-credibility", action="store_true", help="no source tiers, no spam demotion")
    p.add_argument("--no-recency", action="store_true", help="no recency boost or rule")
    p.add_argument("--no-sanitize", action="store_true", help="pass page text to the model untouched")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lab", description="Module 08 lab: the Ask-the-Web agent (offline).")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="search the offline web snapshot")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    add_guard_flags(p)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("open", help="open one page the way the agent sees it")
    p.add_argument("url")
    add_guard_flags(p)
    p.set_defaults(fn=cmd_open)

    for name, fn, help_text in (
        ("ask", cmd_ask, "ask any question"),
        ("task", cmd_task, "run one task from tasks.jsonl and grade it"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("question" if name == "ask" else "task_id")
        p.add_argument("--agent", default="react", choices=list(AGENTS))
        p.add_argument("--trace", action="store_true", help="print every step")
        add_agent_flags(p)
        add_guard_flags(p)
        p.set_defaults(fn=fn)

    p = sub.add_parser("eval", help="run one agent on all 24 tasks")
    p.add_argument("--agent", default="react", choices=list(AGENTS))
    p.add_argument("--save", help="write JSONL records (for example outputs/react.jsonl)")
    add_agent_flags(p)
    add_guard_flags(p)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("compare", help="every agent side by side")
    add_guard_flags(p)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("ablate", help="switch guards off one at a time")
    p.add_argument("--agent", default="react", choices=list(AGENTS))
    p.set_defaults(fn=cmd_ablate)

    p = sub.add_parser("judge", help="compare exact match, lenient match and an LLM judge")
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("replay", help="re-grade a saved run without calling anything")
    p.add_argument("path")
    p.set_defaults(fn=cmd_replay)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
