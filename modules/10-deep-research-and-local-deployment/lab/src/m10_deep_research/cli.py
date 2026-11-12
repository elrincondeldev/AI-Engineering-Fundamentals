"""Command line for the module 10 lab: ``uv run lab --help``."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .credibility import score_page
from .evaluate import ABLATIONS, load_workspace, run_ablations, run_and_grade, with_overrides
from .grade import grade_report
from .loop import ResearchConfig
from .memory import bits_per_weight, decode_ceiling, estimate, kv_bytes_per_token, sweep
from .providers import get_provider
from .quantize import compare
from .serving import (
    CostInputs,
    daily_cost,
    kv_plans,
    make_requests,
    servers_needed,
    simulate_continuous,
    simulate_static,
    speculative,
)

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"


def _config(args: argparse.Namespace) -> ResearchConfig:
    return with_overrides(
        ResearchConfig(),
        use_credibility=False if args.no_credibility else None,
        use_recency=False if args.no_recency else None,
        use_sanitiser=False if args.no_sanitiser else None,
        use_dedup=False if args.no_dedup else None,
        max_iterations=args.max_iterations,
        max_searches=args.max_searches,
        top_k=args.top_k,
    )


def _provider(args: argparse.Namespace):
    return None if args.provider == "fake" else get_provider(args.provider)


def _qids(ws, which: str) -> list[str]:
    return sorted(ws.questions) if which == "all" else [which]


def cmd_research(args: argparse.Namespace) -> None:
    ws = load_workspace()
    OUTPUTS.mkdir(exist_ok=True)
    for qid in _qids(ws, args.question):
        result, grade = run_and_grade(ws, qid, _config(args), _provider(args))
        (OUTPUTS / f"{qid}.md").write_text(result.report, encoding="utf-8")
        (OUTPUTS / f"{qid}.json").write_text(
            json.dumps({**result.summary(), "trace": result.trace}, indent=2), encoding="utf-8")
        if args.trace:
            print("\n".join(result.trace))
        s = result.stats
        print(f"{qid}: {len(result.subqueries)} sub-queries, {s.iterations} iteration(s), "
              f"{s.searches} searches, {s.pages_opened} pages opened, {s.llm_calls} LLM calls")
        print(f"     LLM tokens: {s.prompt_tokens} prompt + {s.completion_tokens} completion")
        print(f"     tokens read {s.tokens_read} -> noted {s.tokens_noted} -> kept {s.tokens_kept}; "
              f"stop: {s.stop_reason}")
        print(f"     notes {len(result.notes)}, gaps {result.gaps or 'none'}; "
              f"facts {grade.facts_covered}/{grade.facts_total}")
        print(f"     report written to outputs/{qid}.md")


def cmd_grade(args: argparse.Namespace) -> None:
    ws = load_workspace()
    print(f"{'question':9}{'facts':>7}{'supported':>11}{'valid cites':>13}{'domains':>9}{'low-cred':>10}"
          f"{'traps':>7}{'PII':>5}")
    for qid in _qids(ws, args.question):
        _, g = run_and_grade(ws, qid, _config(args), _provider(args))
        print(f"{qid:9}{g.facts_covered:>4}/{g.facts_total:<2}{g.facts_supported:>9}/{g.facts_total:<2}"
              f"{g.valid_cited_lines:>9}/{g.cited_lines:<3}{g.distinct_domains:>5}/{g.min_sources:<3}"
              f"{g.low_credibility_citations:>8}{len(g.trap_hits):>7}{g.pii_leaks:>5}"
              + (f"   traps: {', '.join(g.trap_hits)}" if g.trap_hits else ""))


def cmd_grade_file(args: argparse.Namespace) -> None:
    ws = load_workspace()
    g = grade_report(Path(args.path).read_text(encoding="utf-8"), ws.questions[args.question], ws.snapshot,
                     ws.labels)
    print(f"{args.question}: facts {g.facts_covered}/{g.facts_total}, "
          f"supported {g.facts_supported}/{g.facts_total}, "
          f"valid cited lines {g.valid_cited_lines}/{g.cited_lines}, "
          f"domains {g.distinct_domains}/{g.min_sources}, low-cred {g.low_credibility_citations}, "
          f"traps {g.trap_hits or 'none'}, PII {g.pii_leaks}")


def cmd_ablate(args: argparse.Namespace) -> None:
    ws = load_workspace()
    print(f"{'configuration':29}{'facts':>7}{'valid cites':>13}{'low-cred':>10}{'traps':>7}{'PII':>5}"
          f"{'enough src':>12}{'tokens kept':>13}")
    for r in run_ablations(ws, args.only):
        print(f"{r.name:29}{r.facts_covered:>4}/{r.facts_total:<2}{r.valid_cited_lines:>9}/{r.cited_lines:<3}"
              f"{r.low_credibility_citations:>8}{r.trap_hits:>7}{r.pii_leaks:>5}"
              f"{r.questions_with_enough_sources:>10}/4{r.tokens_kept:>13}")


def cmd_search(args: argparse.Namespace) -> None:
    ws = load_workspace()
    for hit in ws.index.search(args.query, k=args.k):
        page = ws.snapshot.get(hit.url)
        cred = score_page(page)
        print(f"{hit.score:7.2f}  cred {cred.score:.1f}  {hit.url}")
        print(f"         {page.published or 'undated'} · {'; '.join(cred.reasons)}")


def cmd_memory(args: argparse.Namespace) -> None:
    if args.sweep:
        print(f"Fits in {args.usable:.0%} of {args.ram:g} GB (context {args.context}, example shape "
              f"{args.layers} layers x {args.kv_heads} KV heads x {args.head_dim} dims):")
        print(f"{'params':>8}{'16-bit':>10}{'8-bit':>10}{'4-bit':>10}")
        rows = sweep(args.ram, usable=args.usable, context=args.context, layers=args.layers,
                     kv_heads=args.kv_heads, head_dim=args.head_dim)
        for p in dict.fromkeys(r.params for r in rows):
            cells = [f"{r.total_gb:.1f}{'' if r.fits else '✗':1}" for r in rows if r.params == p]
            print(f"{p / 1e9:>7g}B" + "".join(f"{c:>10}" for c in cells))
        print("(GB, total; ✗ = does not fit)")
        return
    group = args.group_size if args.bits < 16 else None
    e = estimate(args.params, args.bits, args.context, args.layers, args.kv_heads, args.head_dim,
                 batch=args.batch, kv_bits=args.kv_bits, group_size=group, overhead=args.overhead)
    per_tok = kv_bytes_per_token(args.layers, args.kv_heads, args.head_dim, args.kv_bits)
    print(f"model: {args.params / 1e9:g}B parameters at {args.bits}-bit "
          f"({bits_per_weight(args.bits, group):g} bits per weight with scales)")
    print(f"weights   {e.weights_gb:>7.2f} GB")
    print(f"KV cache  {e.kv_cache_gb:>7.2f} GB  ({per_tok / 1024:g} KiB per token x {args.context} tokens "
          f"x batch {args.batch})")
    print(f"overhead  {e.overhead_gb:>7.2f} GB  ({args.overhead:.0%} planning margin)")
    print(f"total     {e.total_gb:>7.2f} GB")
    if args.bandwidth:
        print(f"decode ceiling {decode_ceiling(args.bandwidth, e.weights_gb):g} tokens/s for one user "
              f"at {args.bandwidth:g} GB/s memory bandwidth (upper bound)")
    if args.ram:
        budget = args.usable * args.ram
        print(f"{'fits' if e.total_gb <= budget else 'does NOT fit'} in {budget:g} GB "
              f"({args.usable:.0%} of {args.ram:g} GB RAM)")


def cmd_quantize(args: argparse.Namespace) -> None:
    print("Seeded 256 x 1024 weight matrix with uneven channels and 0.1% outliers")
    print(f"{'scheme':18}{'bits/weight':>12}{'weight err':>12}{'output err':>12}{'zeroed':>9}")
    for r in compare(seed=args.seed):
        print(f"{r.name:18}{r.bits_per_weight:>12.3f}{r.weight_error:>12.4f}{r.output_error:>12.4f}"
              f"{r.zero_share:>9.1%}")


def cmd_serve_sim(args: argparse.Namespace) -> None:
    reqs = make_requests(n=args.requests, mean_gap_ms=args.gap_ms)
    print(f"{len(reqs)} requests, one every {args.gap_ms:g} ms on average (simulated time, not a benchmark)")
    print(f"{'scheduler':24}{'TTFT p50':>11}{'TTFT p95':>11}{'TPOT':>8}{'tokens/s':>10}")
    runs = (simulate_continuous(reqs, 1), simulate_static(reqs, args.batch),
            simulate_continuous(reqs, args.batch))
    for r in runs:
        name = "no batching" if r.scheduler.endswith("(batch 1)") else r.scheduler
        print(f"{name:24}{r.ttft_p50_ms:>9.0f}ms{r.ttft_p95_ms:>9.0f}ms{r.tpot_mean_ms:>6.1f}ms"
              f"{r.throughput_tok_s:>10.1f}")


def cmd_kv_cache(args: argparse.Namespace) -> None:
    reqs = make_requests(n=args.requests)
    per_tok = kv_bytes_per_token(args.layers, args.kv_heads, args.head_dim)
    print(f"KV budget {args.budget_gb:g} GB, {per_tok / 1024:g} KiB per token, "
          f"max length {args.max_seq_len}, page size {args.block} tokens")
    print(f"{'strategy':12}{'reserved':>11}{'used':>9}{'waste':>8}{'fit at once':>13}")
    for p in kv_plans(reqs, args.budget_gb, per_tok, args.max_seq_len, args.block):
        print(f"{p.strategy:12}{p.tokens_reserved:>11,}{p.tokens_used:>9,}{p.waste:>8.1%}"
              f"{p.concurrent_fit:>9}/{len(reqs)}")


def cmd_speculate(args: argparse.Namespace) -> None:
    print(f"{'acceptance':>10}{'draft k':>9}{'tokens/pass':>13}{'speed-up':>10}")
    for alpha in args.alpha:
        tokens, speedup = speculative(alpha, args.k, args.cost_ratio)
        print(f"{alpha:>10.2f}{args.k:>9}{tokens:>13.3f}{speedup:>9.2f}x")


def cmd_cost(args: argparse.Namespace) -> None:
    c = CostInputs(args.tokens_in, args.tokens_out, args.api_in, args.api_out, args.hardware, args.months,
                   args.watts, args.hours, args.kwh)
    r = daily_cost(c)
    print("All prices are inputs you supply; the defaults are placeholders, not real quotes.")
    print(f"API:   {r.api_eur_per_day:>8.2f} EUR/day")
    print(f"local: {r.local_eur_per_day:>8.2f} EUR/day  (hardware {r.local_hardware_eur_per_day:.2f} "
          f"+ energy {r.local_energy_eur_per_day:.2f})")
    if r.breakeven_multiplier < 1:
        print(f"local is already cheaper (break-even at {r.breakeven_multiplier:g}x today's tokens)")
    else:
        print(f"local becomes cheaper at {r.breakeven_multiplier:g}x today's tokens")
    if args.peak_tokens_s:
        n = servers_needed(args.peak_tokens_s, args.server_tokens_s, args.headroom)
        print(f"capacity: {n} server(s) for a peak of {args.peak_tokens_s:g} tokens/s at "
              f"{args.server_tokens_s:g} tokens/s each with {args.headroom:.0%} headroom")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lab", description="Module 10: deep research and local deployment")
    sub = ap.add_subparsers(dest="command", required=True)

    def research_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("question", nargs="?", default="all", help="rq1..rq4 or all")
        p.add_argument("--no-credibility", action="store_true", help="read every page, trust every source")
        p.add_argument("--no-recency", action="store_true", help="ignore publication dates")
        p.add_argument("--no-sanitiser", action="store_true", help="give raw page text to the reader")
        p.add_argument("--no-dedup", action="store_true", help="keep duplicate pages and mirrors")
        p.add_argument("--max-iterations", type=int)
        p.add_argument("--max-searches", type=int)
        p.add_argument("--top-k", type=int, help="pages opened per search")
        p.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "fake"),
                       choices=["fake", "ollama", "openai"], help="default: $LLM_PROVIDER or fake")

    p = sub.add_parser("research", help="run the deep-research loop and write outputs/<id>.md")
    research_flags(p)
    p.add_argument("--trace", action="store_true", help="print every plan, search, skip and note")
    p.set_defaults(func=cmd_research)
    p = sub.add_parser("grade", help="research and grade: key facts, citations, sources, traps")
    research_flags(p)
    p.set_defaults(func=cmd_grade)
    p = sub.add_parser("grade-file", help="grade a report you edited or got from a real model")
    p.add_argument("question", help="rq1..rq4")
    p.add_argument("path", help="Markdown report, e.g. outputs/rq1.md")
    p.set_defaults(func=cmd_grade_file)
    p = sub.add_parser("ablate", help="switch defences off one at a time and compare")
    p.add_argument("--only", nargs="*", choices=list(ABLATIONS))
    p.set_defaults(func=cmd_ablate)
    p = sub.add_parser("search", help="BM25 search over the snapshot, with credibility scores")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("memory", help="estimate memory: weights + KV cache + overhead")
    p.add_argument("--params", type=float, default=8e9)
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 6, 8, 16])
    p.add_argument("--group-size", type=int, default=32, help="weights per scale (quantised models)")
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--kv-bits", type=int, default=16)
    p.add_argument("--overhead", type=float, default=0.10)
    p.add_argument("--ram", type=float, default=0.0, help="machine RAM in GB")
    p.add_argument("--usable", type=float, default=0.75, help="share of RAM you can give the model")
    p.add_argument("--bandwidth", type=float, default=0.0, help="memory bandwidth in GB/s (speed ceiling)")
    p.add_argument("--sweep", action="store_true", help="table of sizes x bit widths")
    p.set_defaults(func=cmd_memory)
    p = sub.add_parser("quantize", help="int8/int4 error: per-tensor vs per-channel vs per-group")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("serve-sim", help="simulate no batching, static and continuous batching")
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--gap-ms", type=float, default=1500.0)
    p.add_argument("--batch", type=int, default=16)
    p.set_defaults(func=cmd_serve_sim)
    p = sub.add_parser("kv-cache", help="contiguous vs paged KV-cache allocation")
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--budget-gb", type=float, default=8.0)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.set_defaults(func=cmd_kv_cache)
    p = sub.add_parser("speculate", help="expected speed-up of speculative decoding")
    p.add_argument("--alpha", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--cost-ratio", type=float, default=0.05)
    p.set_defaults(func=cmd_speculate)
    p = sub.add_parser("cost", help="API vs local cost per day, and servers for a peak")
    p.add_argument("--tokens-in", type=float, default=1e6, help="input tokens per day")
    p.add_argument("--tokens-out", type=float, default=2e5, help="output tokens per day")
    p.add_argument("--api-in", type=float, default=1.0, help="EUR per million input tokens (placeholder)")
    p.add_argument("--api-out", type=float, default=4.0, help="EUR per million output tokens (placeholder)")
    p.add_argument("--hardware", type=float, default=2500.0, help="EUR for the machine (placeholder)")
    p.add_argument("--months", type=int, default=36, help="amortisation period")
    p.add_argument("--watts", type=float, default=150.0, help="power draw while busy (placeholder)")
    p.add_argument("--hours", type=float, default=8.0, help="busy hours per day")
    p.add_argument("--kwh", type=float, default=0.19, help="EUR per kWh (Spain, Hearthwise dataset)")
    p.add_argument("--peak-tokens-s", type=float, default=0.0)
    p.add_argument("--server-tokens-s", type=float, default=150.0)
    p.add_argument("--headroom", type=float, default=0.3)
    p.set_defaults(func=cmd_cost)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
