"""Command line for the module 07 lab: ``uv run lab <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from . import workflows as wf
from .agency import LADDER_REQUEST, ladder
from .executor import ToolExecutor
from .loop import LoopTrace, run_tool_loop
from .mcp_client import SERVER_CMD, StdioClient, diff_pins, lint_description, pin_tools, remote_tools
from .mcp_server import main as server_main
from .providers import ProviderError, ToolCall, count_tokens, get_provider
from .scripted import scripted_provider
from .store import DEMO_CUSTOMER, OrderStore
from .tools import Session, build_toolbox, inject_faults

PATTERNS = ["chain", "route", "section", "vote", "reflect", "orchestrate"]
DEMO_TIMEOUT_S = 0.2  # tool timeout in the timeout and idempotency demos
DEMO_SLOW_S = 0.5  # how long the injected "slow" faults sleep
DEMOS = ["compare", "fix-args", "return", "return-denied", "not-yours", "retries", "timeout", "idempotency"]


def _short(text: str, width: int = 104) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


# --------------------------------------------------------------------------- #
# ladder                                                                      #
# --------------------------------------------------------------------------- #


def cmd_ladder(_: argparse.Namespace) -> None:
    print(f"Request: {LADDER_REQUEST}\n")
    print(f"{'Level':<30}{'Decides':<11}{'LLM calls':>10}{'Tools':>7}{'Tokens':>8}  Data changed")
    for r in ladder():
        print(f"{r.level:<30}{r.who_decides:<11}{r.llm_calls:>10}{r.tool_calls:>7}{r.tokens:>8}  "
              f"{'yes' if r.data_changed else 'no'}")
    print()
    for r in ladder():
        print(f"[{r.level}] {_short(r.answer, 140)}")


# --------------------------------------------------------------------------- #
# workflow                                                                    #
# --------------------------------------------------------------------------- #


def _usage(report: dict[str, Any]) -> str:
    return (f"LLM calls: {report['llm_calls']} | sequential rounds: {report.get('sequential_rounds', '-')} | "
            f"tokens: {report['prompt_tokens'] + report['completion_tokens']}")


def cmd_workflow(args: argparse.Namespace) -> None:
    patterns = PATTERNS if args.pattern == "all" else [args.pattern]
    for p in patterns:
        print(f"== {p} ==")
        WORKFLOW_PRINTERS[p]()
        print()


def _print_chain() -> None:
    report = wf.chain()
    for msg, res in zip(wf.CUSTOMER_MESSAGES, report["results"]):
        print(f"- {_short(msg, 70):<70} -> {res['outcome']}")
        print(f"    {_short(res['reply'])}")
    print(f"Outcomes: {report['outcomes']}")
    print(_usage(report))


def _print_route() -> None:
    raw, clean = wf.route_tickets(strip=False), wf.route_tickets()
    print(f"Tickets: {clean['tickets']}")
    print(f"Without stripping quoted replies: accuracy {raw['accuracy']:.3f}, to human {raw['lanes']['human']}")
    print(f"With stripping quoted replies:    accuracy {clean['accuracy']:.3f}, to human {clean['lanes']['human']}")
    print(f"Lanes: {clean['lanes']}")
    print(f"Accuracy on tickets routed to a lane: {clean['accuracy_when_routed']:.3f}")
    print(f"Model tiers: {clean['tiers']}")
    print(_usage(clean))


def _print_section() -> None:
    report = wf.section_compare()
    print(report["table"])
    print(f"Sections: {report['sections']} | sequential rounds: {report['sequential_rounds']} "
          f"(would be {report['sequential_rounds_if_serial']} one after another)")
    print(_usage(report))


def _print_vote() -> None:
    clean = wf.route_tickets()
    print(f"Noise-free router accuracy (the ceiling): {clean['accuracy']:.3f}")
    for k in (1, 3, 5):
        r = wf.vote_tickets(k)
        print(f"k={k} votes, noise {r['noise']}: accuracy {r['accuracy']:.3f}, LLM calls {r['llm_calls']}")


def _print_reflect() -> None:
    report = wf.reflect()
    for h in report["history"]:
        print(f"Round {h['round']} draft: {_short(h['draft'], 140)}")
        for issue in h["issues"] or ["none: passed"]:
            print(f"    issue: {issue}")
    print(f"Rounds: {report['rounds']} | passed: {report['passed']}")
    print(_usage(report))


def _print_orchestrate() -> None:
    for request in wf.ORCHESTRATE_REQUESTS:
        r = wf.orchestrate(request)
        print(f"Request: {request}")
        print(f"  first plan: {r['plan']} | workers run: {r['workers']}")
        print(f"  {r['answer']}")
        print(f"  {_usage(r)}")


WORKFLOW_PRINTERS: dict[str, Callable[[], None]] = {
    "chain": _print_chain, "route": _print_route, "section": _print_section, "vote": _print_vote,
    "reflect": _print_reflect, "orchestrate": _print_orchestrate,
}


# --------------------------------------------------------------------------- #
# tools                                                                       #
# --------------------------------------------------------------------------- #


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def print_trace(trace: LoopTrace, approvals: dict[str, bool] | None = None) -> None:
    """Print each model turn, the tool calls it asked for and what happened to them."""
    results = {r.call_id: r for r in trace.results}
    approvals = approvals or {}
    n = 0
    for msg in trace.messages:
        if msg.role != "assistant":
            continue
        n += 1
        if not msg.tool_calls:
            print(f"  LLM call {n} -> answer: {_short(msg.content, 120)}")
            continue
        for tc in msg.tool_calls:
            r = results[tc.id]
            print(f"  LLM call {n} -> {tc.name}({json.dumps(tc.arguments)})")
            if tc.id in approvals:
                print(f"      [approval] a person {'approved' if approvals[tc.id] else 'rejected'} this call")
            detail = f"({_plural(r.attempts, 'attempt')})" if r.ok else _short(r.error, 96)
            print(f"      {r.status} {detail}")
    print(f"  Totals: {_plural(trace.llm_calls, 'LLM call')}, {_plural(len(trace.results), 'tool call')}, "
          f"{trace.prompt_tokens} prompt tokens, {trace.completion_tokens} completion tokens")


DEMO_REQUESTS = {
    "compare": "Compare the Ember T3 and the Lookout C2, please.",
    "fix-args": "Where is my order 310108?",
    "return": "Please return the Ember T3 from order HW-310042, I changed my mind.",
    "return-denied": "Please return the Ember T3 from order HW-310042, I changed my mind.",
    "not-yours": "Where is order HW-418230?",
    "retries": "Where is order HW-310108?",
    "timeout": "How long do refunds take after a return?",
    "idempotency": "Please return one Spark Mini from order HW-310155, I changed my mind.",
}


def run_demo(name: str, keys: bool = True, live: bool = False) -> dict[str, Any]:
    """Run one tool-calling demo, print its trace and return what the tests need.

    ``live=True`` replaces the scripted model with ``get_provider()`` (``LLM_PROVIDER``): untested here.
    """
    store = OrderStore()
    tools = build_toolbox(store)
    approvals: dict[str, bool] = {}
    approver = None
    if name in ("return", "idempotency", "return-denied"):
        approver = _approver(name != "return-denied", approvals)
    if name == "retries":
        tools = inject_faults(tools, {"lookup_order": ["transient", "transient"]})
    if name == "timeout":
        tools = inject_faults(build_toolbox(store, timeout_s=DEMO_TIMEOUT_S), {"search_help_centre": ["slow"]},
                              slow_s=DEMO_SLOW_S)
    if name == "idempotency":
        tools = inject_faults(build_toolbox(store, timeout_s=DEMO_TIMEOUT_S),
                              {"create_return": ["slow_after_commit"]}, slow_s=DEMO_SLOW_S)
    executor = ToolExecutor(tools, Session(DEMO_CUSTOMER), approver=approver, use_idempotency_keys=keys)
    print(f"Request: {DEMO_REQUESTS[name]}")
    provider = get_provider() if live else scripted_provider(executor.specs())
    trace = run_tool_loop(provider, executor, DEMO_REQUESTS[name])
    print_trace(trace, approvals)
    print(f"  Returns in the store: {len(store.returns)}")
    return {"trace": trace, "store": store}


def _approver(decision: bool, log: dict[str, bool]) -> Callable[[ToolCall], bool]:
    """Stands in for a person clicking Approve or Reject in the app."""

    def approver(call: ToolCall) -> bool:
        log[call.id] = decision
        return decision

    return approver


def cmd_tools(args: argparse.Namespace) -> None:
    tools = build_toolbox()
    if not args.demo:
        print(f"{'Tool':<20}{'Changes data':<14}{'Required arguments':<36}{'Spec tokens':>11}")
        for t in tools.values():
            req = ", ".join(t.parameters.get("required", []))
            print(f"{t.name:<20}{'yes' if t.side_effect else 'no':<14}{req:<36}"
                  f"{count_tokens(json.dumps(t.spec())):>11}")
        total = count_tokens(json.dumps([t.spec() for t in tools.values()]))
        print(f"All four specs together: {total} tokens, sent with every LLM call.")
        return
    for name in DEMOS if args.demo == "all" else [args.demo]:
        print(f"== {name} ==")
        run_demo(name, live=args.live)
        if name == "idempotency":
            print("-- the same slow commit, with idempotency keys switched off --")
            run_demo(name, keys=False, live=args.live)
        print()


# --------------------------------------------------------------------------- #
# mcp-demo                                                                    #
# --------------------------------------------------------------------------- #


def _print_transcript(client: StdioClient, start: int) -> None:
    for direction, msg in client.transcript[start:]:
        print(f"  {direction} {_short(json.dumps(msg, ensure_ascii=False), 110)}")


def cmd_mcp_demo(args: argparse.Namespace) -> None:
    approved = pin_tools([t.mcp_spec() for t in build_toolbox().values()])  # recorded at review time
    command = SERVER_CMD + (["--poisoned"] if args.poisoned else [])
    print(f"Host launches: python -m m07_agents_tools.mcp_server{' --poisoned' if args.poisoned else ''} "
          f"(HEARTHWISE_CUSTOMER_ID={DEMO_CUSTOMER})")
    with StdioClient(command=command, env={"HEARTHWISE_CUSTOMER_ID": DEMO_CUSTOMER}) as client:
        client.initialize()
        listed = client.list_tools()
        print("1) Handshake and discovery")
        _print_transcript(client, 0)
        names = {t["name"] for t in listed}
        print("2) Host review: lint descriptions and compare pins with the approved ones")
        for t in listed:
            print(f"  {t['name']:<20} findings: {lint_description(t, names) or 'none'}")
        diff = diff_pins(approved, pin_tools(listed))
        print(f"  pin diff: {diff}")
        blocked = set(diff["added"]) | set(diff["changed"])
        allowed = [t for t in listed if t["name"] not in blocked]
        print(f"  exposed to the model: {[t['name'] for t in allowed]} | quarantined: {sorted(blocked)}")
        if args.poisoned:
            return
        print("3) Direct tools/call requests (identity comes from the launch environment)")
        for order_id in ("HW-310042", "HW-418230"):
            result = client.call_tool("lookup_order", {"order_id": order_id})
            print(f"  -> tools/call lookup_order {{\"order_id\": \"{order_id}\"}}")
            print(f"  <- isError={str(result['isError']).lower()} {_short(result['content'][0]['text'], 90)}")
        print("4) The same tool loop as lesson 3, with tools served over MCP")
        approvals: dict[str, bool] = {}
        executor = ToolExecutor(remote_tools(client, allowed), Session(DEMO_CUSTOMER),
                                approver=_approver(True, approvals))
        request = DEMO_REQUESTS["return"]
        print(f"Request: {request}")
        print_trace(run_tool_loop(scripted_provider(executor.specs()), executor, request), approvals)
        print(f"JSON-RPC messages exchanged: {len(client.transcript)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab", description="Module 07 lab: workflows, tools and MCP.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("ladder", help="one request at four levels of agency").set_defaults(func=cmd_ladder)
    p = sub.add_parser("workflow", help="run a workflow pattern")
    p.add_argument("pattern", choices=PATTERNS + ["all"])
    p.set_defaults(func=cmd_workflow)
    p = sub.add_parser("tools", help="list the toolbox, or run a tool-calling demo")
    p.add_argument("--demo", choices=DEMOS + ["all"])
    p.add_argument("--live", action="store_true", help="use LLM_PROVIDER instead of the scripted model")
    p.set_defaults(func=cmd_tools)
    p = sub.add_parser("mcp-demo", help="launch the teaching MCP server and talk to it")
    p.add_argument("--poisoned", action="store_true", help="use a server with a poisoned tool description")
    p.set_defaults(func=cmd_mcp_demo)
    p = sub.add_parser("mcp-server", help="run the teaching MCP server on stdin/stdout")
    p.add_argument("--poisoned", action="store_true")
    p.set_defaults(func=lambda a: server_main(["--poisoned"] if a.poisoned else []))
    args = ap.parse_args(argv)
    try:
        args.func(args)
    except ProviderError as exc:  # for example --live with no model server running
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
