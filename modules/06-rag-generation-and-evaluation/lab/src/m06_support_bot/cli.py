"""Command line for the module 06 lab: ``uv run lab --help``."""

from __future__ import annotations

import argparse
import sys
from statistics import mean

from .bot import ABLATIONS, BotConfig, SupportBot
from .evaluate import Summary, by_category, check_gates, evaluate, sweep_thresholds
from .judge import KINDS, build_cases, calibrate, heuristic_judge
from .kb import load_documents
from .raft import RaftConfig, build_raft, stats, validate, write_jsonl


def _config(args: argparse.Namespace) -> BotConfig:
    cfg = ABLATIONS[args.config]
    if getattr(args, "threshold", None) is not None:
        cfg = cfg.but(threshold=args.threshold, name=f"{cfg.name}@{args.threshold:.2f}")
    if getattr(args, "top_k", None) is not None:
        cfg = cfg.but(top_k=args.top_k, name=f"{cfg.name}-k{args.top_k}")
    return cfg


def cmd_ask(args: argparse.Namespace) -> int:
    bot = SupportBot(_config(args))
    out = bot.ask(args.question)
    print(f"Question     : {out.question}")
    if out.redactions:
        print(f"Redacted     : {out.redacted_question}")
    print(f"Language     : {out.language}")
    print(f"Route        : {out.route}")
    if out.search_terms:
        print(f"Search terms : {' '.join(out.search_terms)}")
    for i, c in enumerate(out.context, 1):
        label = "Context      :" if i == 1 else "              "
        print(f"{label} {i}. {c.chunk_id} {c.doc.title} > {c.section} [{c.doc.status}, {c.doc.audience}]")
    if out.evidence_score is not None:
        print(f"Evidence     : {out.evidence_score:.3f} (threshold {bot.config.threshold:.2f})")
    print(f"Guardrails   : redactions={','.join(out.redactions) or 'none'} "
          f"output={','.join(out.guardrail) or 'ok'}")
    print(f"Outcome      : {out.reason}")
    print(f"Answer       : {out.answer}")
    print(f"Citations    : {', '.join(out.citations) or '-'}")
    if args.show_prompt and out.messages:
        print("\n--- prompt sent to the model ---")
        for m in out.messages:
            print(f"[{m.role}]\n{m.content}\n")
    return 0


def cmd_sanitize(args: argparse.Namespace) -> int:
    docs = {d.id: d for d in load_documents(sanitize=True)}
    naive = {d.id: d for d in load_documents(sanitize=False)}
    targets = [args.doc_id] if args.doc_id else [i for i, d in docs.items() if d.removed]
    for doc_id in targets:
        d = docs[doc_id]
        print(f"{doc_id} {d.path} (source={d.source})")
        print(f"  words seen by a naive parser : {len(naive[doc_id].text.split())}")
        print(f"  words kept by the sanitiser  : {len(d.text.split())}")
        for frag in d.removed:
            print(f"  removed: {frag}")
    return 0


def _rate(n: int, d: int) -> str:
    return f"{n}/{d} = {n / d:.3f}" if d else "-"


def print_summary(s: Summary) -> None:
    rows = [
        ("answer correctness (all key facts)", _rate(s.correct, s.answerable)),
        ("key-fact recall (mean)", f"{s.fact_recall:.3f}"),
        ("token F1 vs reference (mean)", f"{s.token_f1:.3f}"),
        ("faithfulness (claims supported)", f"{s.faithfulness:.3f}"),
        ("context relevance (precision@k)", f"{s.context_relevance:.3f}"),
        ("context recall (hit@k)", f"{s.context_recall:.3f}"),
        ("MRR@k", f"{s.mrr:.3f}"),
        ("abstention accuracy", _rate(round(s.abstention_accuracy * s.questions), s.questions)),
        ("unanswerable questions refused", f"{s.unanswerable_abstained}/{s.unanswerable}"),
        ("false abstentions (answerable)", str(s.false_abstentions)),
        ("must_not_contain violations", str(s.violations)),
        ("global deny-list leaks", str(s.leaks)),
        ("guardrail blocks", str(s.guardrail_blocks)),
        ("language match (answered)", f"{s.language_match:.3f}"),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{k:<{width}}  {v}")


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _config(args)
    summary, results = evaluate(cfg, write=True)
    print(f"Golden set: {summary.questions} questions ({summary.answerable} answerable, "
          f"{summary.unanswerable} must be refused) | config: {cfg.name}\n")
    print_summary(summary)
    print("\ncategory            n  answered  correct  abst.ok  violations")
    for row in by_category(results):
        print(f"{row['category']:<18} {row['n']:>2}  {row['answered']:>8}  {row['correct']:>7}  "
              f"{row['abstention_ok']:>7}  {row['violations']:>10}")
    problems = [r for r in results if (r.answerable and not r.correct) or not r.abstention_ok
                or r.violations or r.leaks or r.language_match is False]
    if problems:
        print("\nTo review:")
        for r in problems:
            facts = f"facts={r.fact_recall:.2f}" if r.fact_recall is not None else "facts=-"
            lang = " lang-mismatch" if r.language_match is False else ""
            print(f"  {r.id} {r.category:<14} {r.reason:<9} {facts}{lang} | {r.answer[:70]}")
    print(f"\nPer-question details: outputs/eval_{cfg.name}.jsonl")
    if args.gate:
        failed = check_gates(summary)
        print("Regression gate: " + ("PASS" if not failed else "FAIL " + "; ".join(failed)))
        return 1 if failed else 0
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    print("config                     correct  abst.acc  violations  leaks  blocked  leaked ids")
    for name in ABLATIONS:
        s, _ = evaluate(ABLATIONS[name])
        ids = ",".join(sorted(set(s.violation_ids) | set(s.leak_ids))) or "-"
        print(f"{name:<26} {s.correct:>3}/{s.answerable:<3}  {s.abstention_accuracy:>8.3f}  "
              f"{s.violations:>10}  {s.leaks:>5}  {s.guardrail_blocks:>7}  {ids}")
    print("\nthreshold  answered  correct  unanswerable refused  false abstentions  abst.acc")
    for s in sweep_thresholds([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
        t = s.config.split("=")[1]
        print(f"{t:>9}  {s.answered:>8}  {s.correct:>7}  {s.unanswerable_abstained:>18}/{s.unanswerable}  "
              f"{s.false_abstentions:>17}  {s.abstention_accuracy:>8.3f}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    cases = build_cases(seed=args.seed)
    counts = {k: sum(c.kind == k for c in cases) for k in KINDS}
    print(f"Judge cases: {len(cases)} ({', '.join(f'{k} {v}' for k, v in counts.items())})")
    print("Labels are known by construction: original and paraphrase are faithful, the rest are not.\n")
    print(f"{'judge':<22} {'agreement':>9} {'false alarms':>12} {'number_swap':>11} {'foreign_claim':>13} "
          f"{'negation':>8}")
    for tau in (0.5, 0.7, 0.9):
        for numbers in (False, True):
            cal = calibrate(f"tau={tau} numbers={'on' if numbers else 'off'}",
                            heuristic_judge(tau, numbers), cases)
            ns, fc, ng = (f"{cal.caught[k][0]}/{cal.caught[k][1]}" for k in ("number_swap", "foreign_claim",
                                                                              "negation"))
            fa = f"{cal.false_alarms}/{cal.faithful_cases}"
            print(f"{cal.name:<22} {cal.agreement:>9.3f} {fa:>12} {ns:>11} {fc:>13} {ng:>8}")
    return 0


def cmd_raft(args: argparse.Namespace) -> int:
    cfg = RaftConfig(p_oracle=args.p_oracle, copies=args.copies, n_docs=args.docs, seed=args.seed,
                     no_oracle_target=args.no_oracle_target, include_restricted=args.include_restricted)
    examples, train, test = build_raft(cfg)
    st = stats(examples)
    print(f"RAFT dataset | p_oracle={cfg.p_oracle} | {cfg.n_docs} docs per example | {cfg.copies} copies "
          f"| no-oracle target: {cfg.no_oracle_target}")
    print(f"Questions    : {len(train)} train / {len(test)} test (held out for evaluation)")
    print(f"Examples     : {st['examples']} | with oracle: {st['with_oracle']} | "
          f"NO_ANSWER targets: {st['abstain_targets']}")
    print(f"Tokens       : {st['avg_tokens']} per example (approx.) | {st['total_tokens']} in total")
    checks = validate(examples, train, test, cfg)
    print("Checks:")
    for c in checks:
        detail = "" if c.passed else f" ({len(c.failures)} examples, e.g. {', '.join(c.failures[:3])})"
        print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name}{detail}")
    p1 = write_jsonl(examples, "raft_train.jsonl")
    p2 = write_jsonl([{"id": r["id"], "question": r["question"]} for r in test], "raft_test_questions.jsonl")
    print(f"Wrote outputs/{p1.name} and outputs/{p2.name}")
    if args.show is not None:
        ex = examples[args.show]
        print(f"\n--- example {args.show}: {ex['id']} (oracle: {ex['oracle_chunk'] or 'none'}) ---")
        print(f"question : {ex['messages'][1]['content'].rsplit('<question>', 1)[1].split('</question>')[0]}")
        print(f"documents: {', '.join(ex['context_chunk_ids'])}")
        print(ex["messages"][-1]["content"])
    return 0 if all(c.passed for c in checks) else 1


def cmd_budget(args: argparse.Namespace) -> int:
    print("top_k  avg prompt tokens (rag route)  context recall  correct")
    for k in (1, 2, 4, 8):
        s, results = evaluate(BotConfig(name=f"k={k}", top_k=k))
        toks = [r.prompt_tokens for r in results if r.route == "rag" and r.prompt_tokens]
        print(f"{k:>5}  {mean(toks):>29.1f}  {s.context_recall:>14.3f}  {s.correct:>4}/{s.answerable}")
    _, results = evaluate(BotConfig())
    routes = {r: sum(x.route == r for x in results) for r in ("faq", "rag", "human", "safety")}
    print("\nRoutes (top_k=4): " + " | ".join(f"{k} {v}" for k, v in routes.items()))
    print(f"Model calls: {routes['rag']} of {len(results)} questions "
          "(FAQ, human and safety routes skip the model)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab", description="Module 06: the grounded Hearthwise support bot.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask", help="ask the bot one question and show the trace")
    p.add_argument("question")
    p.add_argument("--config", choices=list(ABLATIONS), default="full")
    p.add_argument("--threshold", type=float, default=None, help="abstention threshold (default 0.5)")
    p.add_argument("--top-k", type=int, default=None, help="chunks to retrieve (default 4)")
    p.add_argument("--show-prompt", action="store_true", help="print the exact messages sent to the model")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("sanitize", help="show what the sanitiser removes from the help centre")
    p.add_argument("doc_id", nargs="?", default=None)
    p.set_defaults(func=cmd_sanitize)

    p = sub.add_parser("eval", help="score the bot on the 40 golden questions")
    p.add_argument("--config", choices=list(ABLATIONS), default="full")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--gate", action="store_true", help="exit 1 if the regression gate fails")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("ablate", help="switch defences off one at a time; sweep the abstention threshold")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("judge", help="calibrate the faithfulness judge on labelled corruptions")
    p.add_argument("--seed", type=int, default=6)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("raft", help="build and validate a RAFT-style training set in outputs/")
    p.add_argument("--p-oracle", type=float, default=0.8)
    p.add_argument("--copies", type=int, default=3)
    p.add_argument("--docs", type=int, default=4)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--no-oracle-target", choices=["abstain", "answer"], default="abstain")
    p.add_argument("--include-restricted", action="store_true",
                   help="let internal and superseded documents in (breaks validation on purpose)")
    p.add_argument("--show", type=int, default=None, metavar="N", help="print training example number N")
    p.set_defaults(func=cmd_raft)

    p = sub.add_parser("budget", help="prompt size versus quality for different top_k values")
    p.set_defaults(func=cmd_budget)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
