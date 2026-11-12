"""Command line for the module 04 lab: ``uv run lab <command>``."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

from . import reports
from .evaluation import HEADER, evaluate_prompt, make_provider
from .prompts import ExampleSelector, PromptConfig, build_messages
from .structured import classify
from .tickets import INTENTS


def _title(text: str) -> None:
    print(text)
    print("-" * len(text))


def cmd_audit(_: argparse.Namespace) -> None:
    r = reports.audit_report()
    _title("Ticket audit")
    print(f"rows: {r.rows} · resubmitted copies: {r.resubmissions} · repeated bodies (any customer): "
          f"{r.duplicate_bodies}")
    print(f"after removing resubmissions: {r.after_dedup} · temporal split: train {r.train} "
          f"(before 2025-09-01) / test {r.test}")
    print("personal data and noise in bodies: " + ", ".join(f"{k} {v}" for k, v in r.pii.items()))
    print()
    _title("Can one column predict the intent? (rule learned on train, scored on test)")
    print(f"{'column':<16} {'values':>6} {'accuracy':>9}  known when the ticket arrives?")
    for c in r.columns:
        flag = "yes" if c.known_at_prediction_time else "NO  <- leak: filled in after the ticket is solved"
        print(f"{c.column:<16} {c.distinct_values:>6} {c.accuracy:>9.3f}  {flag}")
    print(f"majority-class baseline on test: {r.majority:.3f}")
    print()
    _title("Drift after the Lookout C2 launch")
    print(f"{'intent':<20} {'before':>7} {'after':>7}")
    for label in INTENTS:
        print(f"{label:<20} {r.shares_before[label]:>7.3f} {r.shares_after[label]:>7.3f}")
    print(f"Lookout C2 share of test tickets: {r.camera_share_after:.3f}")
    print(f"unseen word stems vs Jan-Jul vocabulary: August {r.oov_august:.3f} · after launch "
          f"{r.oov_after_launch:.3f} · camera tickets {r.oov_camera:.3f}")


def cmd_splits(_: argparse.Namespace) -> None:
    _title("Same prompt (bilingual descriptions + 8 similar examples), different evaluation designs")
    print(f"{'split':<30} {'train':>5} {'test':>5} {'overlap':>7} {'accuracy':>8}")
    for row in reports.split_report():
        print(f"{row.name:<30} {row.train:>5} {row.test:>5} {row.overlap:>7} {row.accuracy:>8.3f}")
    print("overlap = test tickets whose exact text also appears in train")


def _config_from(args: argparse.Namespace) -> PromptConfig:
    return PromptConfig(descriptions=args.descriptions, k=args.k, selection=args.selection,
                        output=args.output, reasoning=args.reasoning, user_context=args.user_context)


def cmd_show_prompt(args: argparse.Namespace) -> None:
    train, test = reports.standard_split()
    ticket = next(t for t in test if t.ticket_id == args.ticket)
    cfg = _config_from(args)
    messages = build_messages(ticket, cfg, ExampleSelector(train, cfg))
    for m in messages:
        print(f"[{m.role}]\n{m.content}\n")
    provider = make_provider()
    print(f"[model answer]\n{provider.complete(messages).text}\n")
    result = classify(provider, messages, INTENTS, json_mode=cfg.output == "json")
    print(f"-> predicted: {result.label} · true: {ticket.intent} · attempts: {result.attempts} · "
          f"prompt tokens: {result.prompt_tokens}")


def cmd_prompts(args: argparse.Namespace) -> None:
    provider = os.environ.get("LLM_PROVIDER", "fake")
    _title(f"Prompting strategies on the temporal test set (provider: {provider})")
    print(HEADER)
    if args.limit or provider != "fake":
        train, test = reports.standard_split()
        test = test[: args.limit or 40]
        rows = [evaluate_prompt(name, test, train, cfg) for name, cfg in reports.PROMPT_GRID]
    else:
        rows = reports.prompt_table()
    for r in rows:
        print(r.row())
    print(f"n = {rows[0].n} tickets · tokens = mean prompt tokens per ticket")


def cmd_format(_: argparse.Namespace) -> None:
    f = reports.format_report()
    _title(f"Structured output on {f.n} test tickets (bilingual + 8 similar, JSON)")
    print(f"first answer is strict, valid JSON:        {f.strict_valid}")
    print(f"first answer readable by a lenient parser: {f.lenient_valid}")
    print(f"needed a retry:                            {f.needed_retry}")
    print(f"still invalid after one retry:             {f.invalid_after_retry}")
    print(f"accuracy without retry: {f.accuracy_no_retry:.3f} · with one retry: {f.accuracy_with_retry:.3f}")
    print(f"step-by-step prompt: accuracy {f.accuracy_reasoning:.3f} · completion tokens "
          f"{f.completion_tokens_plain:.1f} -> {f.completion_tokens_reasoning:.1f}")


def cmd_context(args: argparse.Namespace) -> None:
    ticket, a = reports.assembly_for(args.ticket, args.budget, args.policy)
    _title(f"Context for {ticket.ticket_id} · budget {args.budget} tokens · policy {args.policy}")
    print(f"offered {a.tokens_offered} tokens · sent {a.tokens} tokens · examples kept {a.examples_kept}")
    for item in a.kept:
        print(f"  KEEP {item.kind:<8} {item.tokens:>4} tok  priority {item.priority:<7} {item.text[:48]!r}")
    for item, why in a.dropped:
        print(f"  DROP {item.kind:<8} {item.tokens:>4} tok  priority {item.priority:<7} ({why}) {item.text[:30]!r}")


def cmd_budget(_: argparse.Namespace) -> None:
    _title("Token budget sweep (16 candidate examples, profile, query; 400 test tickets)")
    print(f"{'budget':>6} {'policy':<13} {'accuracy':>8} {'examples':>8} {'tokens':>7}")
    for row in reports.budget_table():
        print(f"{row.budget:>6} {row.policy:<13} {row.accuracy:>8.3f} {row.examples:>8.2f} {row.tokens:>7.1f}")
    print()
    _title("More examples are not always better")
    print(HEADER)
    for r in reports.k_sweep():
        print(r.row())
    q = reports.quoted_reply_slice()
    print()
    _title(f"Tickets with a quoted earlier e-mail (n = {q.n})")
    print(f"0-shot: raw {q.zero_shot_raw:.3f} -> cleaned {q.zero_shot_clean:.3f}")
    print(f"8-shot: raw {q.few_shot_raw:.3f} -> cleaned {q.few_shot_clean:.3f}")


def cmd_cache(_: argparse.Namespace) -> None:
    stable, unstable = reports.cache_report()
    _title("Prefix cache simulation over the 400 test requests")
    for name, c in (("stable system prompt", stable), ("request id + date first", unstable)):
        print(f"{name:<24} hits {c.hits:>3} · misses {c.misses:>3} · hit rate {c.hit_rate:.3f} · "
              f"cached share of prompt tokens {c.cached_share:.3f}")


def cmd_drift(_: argparse.Namespace) -> None:
    _title("Prompt-side adaptation (old = August, new = October-December)")
    print(f"{'approach':<32} {'labelled':>8} {'old':>6} {'new':>6} {'camera':>7} {'tokens':>7}")
    for r in reports.prompt_adaptation():
        print(f"{r.name:<32} {r.labelled:>8} {r.old:>6.3f} {r.new:>6.3f} {r.camera:>7.3f} {r.prompt_tokens:>7.1f}")


def cmd_lora(_: argparse.Namespace) -> None:
    _title("Adapting the frozen classifier with 97 September tickets (LoRA: mean of 5 seeds)")
    print(f"{'model':<26} {'trainable':>9} {'share':>6} {'old':>6} {'new':>6} {'camera':>7} {'spread':>7}")
    for r in reports.lora_table():
        print(f"{r.name:<26} {r.trainable:>9} {r.share_of_full:>6.3f} {r.old:>6.3f} {r.new:>6.3f} "
              f"{r.camera:>7.3f} {r.new_spread:>7.3f}")
    print()
    _title("Forgetting: train longer on September only (old = August accuracy)")
    print(f"{'steps':>5} {'full old':>9} {'full new':>9} {'r=4 old':>8} {'r=4 new':>8} {'replay old':>10} "
          f"{'replay new':>10}")
    for f in reports.forgetting_table():
        print(f"{f.steps:>5} {f.full_old:>9.3f} {f.full_new:>9.3f} {f.lora_old:>8.3f} {f.lora_new:>8.3f} "
              f"{f.replay_old:>10.3f} {f.replay_new:>10.3f}")
    print()
    _title("How many labelled September tickets? (accuracy on October-December)")
    print(f"{'labelled':>8} {'full':>6} {'LoRA r=4':>9}")
    for c in reports.data_curve():
        print(f"{c.labelled:>8} {c.full_new:>6.3f} {c.lora_new:>9.3f}")


def cmd_lora_params(args: argparse.Namespace) -> None:
    full = args.d_in * args.d_out
    _title(f"LoRA size for one {args.d_out} x {args.d_in} weight matrix (full: {full:,} numbers)")
    print(f"{'rank':>4} {'LoRA numbers':>13} {'share of full':>14}")
    for row in reports.param_table(args.d_in, args.d_out):
        print(f"{row.rank:>4} {row.lora:>13,} {row.share:>14.5f}")
    print()
    mem = reports.memory_table(args.model_params)
    print(f"weights-only memory for {args.model_params:,.0f} parameters: "
          + " · ".join(f"{bits}-bit {gb} GB" for bits, gb in mem.items()))


COMMANDS: dict[str, tuple[Callable[[argparse.Namespace], None], str]] = {
    "audit": (cmd_audit, "lesson 1: duplicates, personal data, leaky columns and drift"),
    "splits": (cmd_splits, "lesson 1: random versus temporal split, and a leak through the prompt"),
    "show-prompt": (cmd_show_prompt, "lesson 2: print the messages sent for one ticket"),
    "prompts": (cmd_prompts, "lesson 2: zero-shot versus few-shot, example selection, user context"),
    "format": (cmd_format, "lesson 2: JSON validation, retries and step-by-step prompting"),
    "context": (cmd_context, "lesson 3: assemble one prompt under a token budget and show what was dropped"),
    "budget": (cmd_budget, "lesson 3: budget sweep, compaction policies and too many examples"),
    "cache": (cmd_cache, "lesson 3: prefix caching with a stable versus an unstable prefix"),
    "drift": (cmd_drift, "lesson 4: adapt by editing the prompt or refreshing the example pool"),
    "lora": (cmd_lora, "lesson 4: LoRA ranks versus full fine-tuning, forgetting and data needs"),
    "lora-params": (cmd_lora_params, "lesson 4: LoRA parameter counts and weight memory"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab", description="Module 04 lab: adapt an LLM to route tickets.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        if name in ("show-prompt", "context"):
            default = "T00843" if name == "show-prompt" else "T00822"
            p.add_argument("--ticket", default=default, help="a test ticket id (from 2025-09-01 onwards)")
        if name == "show-prompt":
            p.add_argument("--descriptions", default="bilingual", choices=["names", "english", "bilingual",
                                                                           "bilingual_v2"])
            p.add_argument("--k", type=int, default=2)
            p.add_argument("--selection", default="similar", choices=["random", "balanced", "similar"])
            p.add_argument("--output", default="json", choices=["json", "label"])
            p.add_argument("--reasoning", action="store_true")
            p.add_argument("--user-context", action="store_true")
        if name == "context":
            p.add_argument("--budget", type=int, default=500)
            p.add_argument("--policy", default="clean+dedupe", choices=list(reports.POLICIES))
        if name == "prompts":
            p.add_argument("--limit", type=int, default=0, help="only the first N test tickets (real models)")
        if name == "lora-params":
            p.add_argument("--d-in", type=int, default=4096)
            p.add_argument("--d-out", type=int, default=4096)
            p.add_argument("--model-params", type=float, default=7e9)
    return parser


OFFLINE_ONLY = {"splits", "format", "budget", "drift"}  # each replays hundreds of model calls


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider = os.environ.get("LLM_PROVIDER", "fake").strip().lower()
    if args.command in OFFLINE_ONLY and provider != "fake":
        print(f"note: '{args.command}' replays hundreds of model calls, so it always uses the offline toy learner.\n"
              "      With a real model use 'show-prompt' or 'prompts --limit N'.\n")
        os.environ["LLM_PROVIDER"] = "fake"
    COMMANDS[args.command][0](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
