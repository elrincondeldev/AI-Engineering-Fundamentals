"""Command-line entry point: ``uv run lab <command>``. Run ``uv run lab --help`` for the list."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence

from . import decoding, evaluation, reward, sft
from .data import read_jsonl
from .ngram import NgramLM
from .providers import get_provider
from .text import load_corpus, tokenize


def build_lm() -> NgramLM:
    """The base model of this lab: a trigram model of the English Hearthwise corpus."""
    return NgramLM().fit(tokenize(d.text) for d in load_corpus())


# --------------------------------------------------------------------------- #
# Lesson 2.1 — decoding                                                       #
# --------------------------------------------------------------------------- #


def cmd_corpus(_: argparse.Namespace) -> None:
    docs = load_corpus()
    lm = build_lm()
    kb = sum(not d.source.startswith("http") for d in docs)
    print(f"documents: {len(docs)} ({kb} help-centre pages, {len(docs) - kb} web pages)")
    print(f"training tokens (including <eos>): {lm.n_train_tokens}")
    print(f"vocabulary (words seen at least twice + 3 special tokens): {len(lm.vocab)}")


def cmd_next_token(args: argparse.Namespace) -> None:
    lm = build_lm()
    print(f"prompt: {args.prompt!r}")
    for t in (0.5, 1.0, 2.0):
        tops = decoding.top_tokens(lm, args.prompt, temperature=t, n=5)
        print(f"T={t:<4} " + "  ".join(f"{tok!r} {p:.3f}" for tok, p in tops))
    sizes = ", ".join(f"p={p}: {decoding.nucleus_size(lm, args.prompt, p=p)}" for p in (0.5, 0.9))
    print(f"tokens kept by top-p ({sizes}) out of {len(lm.vocab)}")


def cmd_generate(args: argparse.Namespace) -> None:
    lm = build_lm()
    cfg = decoding.DecodeConfig(
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        max_new_tokens=args.max_tokens,
        stop=tuple(args.stop),
        seed=args.seed,
    )
    g = decoding.generate(lm, args.prompt, cfg)
    print(f"{args.prompt} -> {g.text}")
    print(f"[finish_reason={g.finish_reason} tokens={len(g.tokens)} "
          f"logprob={g.logprob:.2f} rep-3={decoding.repetition_rate(g.tokens):.2f}]")


def cmd_decode_bench(_: argparse.Namespace) -> None:
    lm = build_lm()
    rows = decoding.decoding_benchmark(lm)
    print(f"{len(decoding.BENCH_PROMPTS)} prompts x 5 seeds, up to 50 new tokens each")
    print(f"{'strategy':22} {'distinct-2':>10} {'rep-3':>6} {'unique':>7} {'logp/tok':>9} {'ended':>6}")
    for r in rows:
        print(f"{r.name:22} {r.distinct_2:10.3f} {r.rep_3:6.3f} {r.unique_outputs:>4}/{r.n_outputs:<2} "
              f"{r.logprob_per_token:9.2f} {r.eos_rate:6.2f}")


# --------------------------------------------------------------------------- #
# Lesson 2.2 — supervised fine-tuning                                         #
# --------------------------------------------------------------------------- #


def cmd_chat_template(args: argparse.Namespace) -> None:
    rows, _ = sft.load_sft()
    row = next(r for r in rows if r["id"] == args.id)
    rendered = sft.render_chat(sft.row_to_messages(row))
    print(rendered, end="")
    toks = sft.chat_tokens(rendered)
    mask = sft.loss_mask(toks)
    print("--- loss mask ([x] = trained) ---")
    print(" ".join(f"[{t}]" if m else t for t, m in zip(toks, mask)))
    print(f"tokens: {len(toks)}  trained: {sum(mask)}  ({sum(mask) / len(toks):.0%})")


def cmd_sft_clean(_: argparse.Namespace) -> None:
    rows, labels = sft.load_sft()
    report = sft.clean(rows)
    print(f"rows: {len(rows)}  kept: {len(report.kept)}  dropped: {len(report.dropped)}")
    print(f"{'check':24} {'labelled':>8} {'flagged':>8} {'correct':>8} {'precision':>9} {'recall':>7}")
    for s in sft.score_against_labels(report, labels):
        print(f"{s.flag:24} {s.labelled:8} {s.flagged:8} {s.correct:8} {s.precision:9.2f} {s.recall:7.2f}")
    ok, total = sft.duplicates_resolved(report, rows, labels)
    print(f"{'duplicate':24} dropped {report.count('duplicate')} exact repeats; "
          f"labelled duplicates kept exactly once: {ok}/{total}")
    leaks = sft.question_leaks(report.kept)
    print(f"kept rows that still ask an eval question word for word: {len(leaks)}")
    print(f"distinct responses among kept rows: {len(sft.response_repeats(report.kept))}")
    before, after = sft.mask_stats(rows), sft.mask_stats(report.kept)
    print(f"tokens before cleaning: {before.total_tokens} ({before.trained_share:.0%} trained)  "
          f"after: {after.total_tokens} ({after.trained_share:.0%} trained)")


# --------------------------------------------------------------------------- #
# Lesson 2.3 — reward models and preference optimisation                      #
# --------------------------------------------------------------------------- #


def cmd_reward(_: argparse.Namespace) -> None:
    train, test = reward.load_pairs("train"), reward.load_pairs("test")
    print(f"'longer answer wins' baseline: train {reward.chosen_is_longer(train):.3f}  "
          f"test {reward.chosen_is_longer(test):.3f}")
    padding = reward.most_common_sentence(train)
    print(f"padding used for the hack test: {padding!r}")
    tables = reward.reward_experiments()
    titles = {"official": "official split (same prompts in train and test)",
              "new_prompts": "new prompts (train and test questions do not overlap)"}
    for key, rows in tables.items():
        print(f"\n{titles[key]}")
        print(f"{'reward model':26} {'pairs':>5} {'feats':>5} {'train':>6} {'test':>6} {'hack':>6}")
        for r in rows:
            print(f"{r.name:26} {r.n_train_pairs:5} {r.n_features:5} {r.train_acc:6.3f} "
                  f"{r.test_acc:6.3f} {r.hack_rate:6.3f}")
    train_a, _ = reward.prompt_disjoint(train, test)
    feats = reward.Featurizer.build(train_a, use_words=True, use_length=True)
    model = reward.fit_reward_model(train_a, feats)
    pos, neg = model.top_weights(5)
    print("\nwords + length (new prompts), largest weights:")
    print("  + " + ", ".join(f"{w} {v:.2f}" for w, v in pos))
    print("  - " + ", ".join(f"{w} {v:.2f}" for w, v in neg))


def cmd_verify(_: argparse.Namespace) -> None:
    problem = next(r for r in read_jsonl("reasoning/problems.jsonl") if r["id"] == "r001")
    print(f"{problem['id']}: {problem['question']}")
    print(f"gold answer: {problem['answer']}")
    for answer in VERIFY_EXAMPLES:
        print(f"reward {reward.verifiable_reward(answer, problem['answer']):.1f} <- {answer!r}")


VERIFY_EXAMPLES = (
    "206 x 0.85 = 175.10, shipping is free, so the total is 175.10 euros.",
    "The total is €175.1",
    "Subtotal 206, minus 15% is 175.10, plus €4.95 shipping = 180.05",
)


def cmd_dpo(_: argparse.Namespace) -> None:
    print(f"reference model: log p(chosen) = {reward.REF_CHOSEN}, log p(rejected) = {reward.REF_REJECTED}")
    print(f"{'policy scenario':42} {'margin':>6} {'loss b=0.1':>10} {'loss b=0.5':>10}")
    for name, margin, losses in reward.dpo_table((0.1, 0.5)):
        print(f"{name:42} {margin:6.1f} {losses[0]:10.3f} {losses[1]:10.3f}")
    r = reward.kl_penalized_reward(reward=2.0, policy_logp=-12.0, ref_logp=-15.0, beta=0.1)
    print(f"\nPPO-style reward: r=2.0, log pi=-12.0, log pi_ref=-15.0, beta=0.1 -> {r:.2f}")


# --------------------------------------------------------------------------- #
# Lesson 2.4 — evaluation                                                     #
# --------------------------------------------------------------------------- #


def cmd_metrics(_: argparse.Namespace) -> None:
    ref, rows = evaluation.metric_table()
    print(f"reference answer (rag_eval q001): {ref!r}")
    print(f"{'candidate':26} {'correct':>7} {'EM':>5} {'F1':>5} {'BLEU':>5} {'ROUGE-L':>7}")
    for r in rows:
        ok = "yes" if r.correct else "no"
        print(f"{r.name:26} {ok:>7} {r.em:5.2f} {r.f1:5.2f} {r.bleu:5.2f} {r.rouge_l:7.2f}")
    print("\nperplexity of a trigram model trained on 90% of the corpus")
    print(f"{'text':26} {'tokens':>6} {'<unk>':>6} {'perplexity':>10}")
    for p in evaluation.perplexity_report():
        print(f"{p.text:26} {p.tokens:6} {p.unk_share:6.0%} {p.perplexity:10.2f}")


def cmd_contamination(_: argparse.Namespace) -> None:
    rows, bad_pages, truth = evaluation.contamination_sweep()
    print(f"web pages labelled as contaminated: {bad_pages}; eval questions copied word for word: {truth}")
    print(f"{'n':>3} {'share':>5} {'pages flagged':>13} {'correct':>7} {'items flagged':>13} {'correct':>7}")
    for r in rows:
        print(f"{r.n:3} {r.min_share:5.1f} {r.pages_flagged:13} {r.pages_correct:7} "
              f"{r.items_flagged:13} {r.items_correct:7}")
    found = evaluation.find_contamination(evaluation.load_web_texts(), evaluation.load_eval_items(), 8, 1.0)
    for url, hits in found.items():
        print(f"{url}: {', '.join(hits)}")


def cmd_arena(_: argparse.Namespace) -> None:
    votes = evaluation.simulate_votes()
    elo = evaluation.elo_ratings(votes)
    elo_rev = evaluation.elo_ratings(list(reversed(votes)))
    bt = evaluation.bradley_terry_ratings(votes)
    ci = evaluation.bootstrap_bt(votes)
    truth = evaluation.true_elo_scale()
    print(f"{len(votes)} simulated votes between {len(truth)} models")
    print(f"{'model':8} {'truth':>6} {'Elo':>6} {'Elo rev':>7} {'BT':>6} {'BT 95% interval':>16}")
    for m in evaluation.ranking(bt):
        lo, hi = ci[m]
        print(f"{m:8} {truth[m]:6.0f} {elo[m]:6.0f} {elo_rev[m]:7.0f} {bt[m]:6.0f} {lo:8.0f} to {hi:4.0f}")
    print("Elo ranking:          " + " > ".join(evaluation.ranking(elo)))
    print("Elo ranking reversed: " + " > ".join(evaluation.ranking(elo_rev)))
    print("BT ranking:           " + " > ".join(evaluation.ranking(bt)))


def cmd_judge(_: argparse.Namespace) -> None:
    # The fake judge is scripted with known biases; a real model is used when LLM_PROVIDER is set.
    if os.environ.get("LLM_PROVIDER", "fake").strip().lower() == "fake":
        provider = get_provider("fake", responder=evaluation.biased_judge_responder())
    else:
        provider = get_provider()
    print(f"judge provider: {provider.name}")
    test = reward.load_pairs("test")
    sets = {"test pairs as rated": test, "test pairs, filler removed": evaluation.strip_padding(test)}
    print(f"{'pairs':28} {'n':>3} {'agree AB':>8} {'agree BA':>8} {'consistent':>10} "
          f"{'first':>6} {'longer':>6}")
    for name, pairs in sets.items():
        r = evaluation.judge_swap_test(pairs, provider)
        print(f"{name:28} {r.n_pairs:3} {r.agreement_ab:8.3f} {r.agreement_ba:8.3f} {r.consistency:10.3f} "
              f"{r.first_position_rate:6.3f} {r.longer_rate:6.3f}")


# --------------------------------------------------------------------------- #
# Wiring                                                                      #
# --------------------------------------------------------------------------- #

COMMANDS: dict[str, tuple[str, Callable[[argparse.Namespace], None]]] = {
    "corpus": ("2.1 size of the training corpus and vocabulary", cmd_corpus),
    "next-token": ("2.1 next-token probabilities at three temperatures", cmd_next_token),
    "generate": ("2.1 generate text with any decoding strategy", cmd_generate),
    "decode-bench": ("2.1 diversity and repetition of every strategy", cmd_decode_bench),
    "chat-template": ("2.2 render one SFT row and show its loss mask", cmd_chat_template),
    "sft-clean": ("2.2 clean the SFT data and score it against the labels", cmd_sft_clean),
    "reward": ("2.3 train Bradley-Terry reward models and expose the length trap", cmd_reward),
    "verify": ("2.3 a verifiable reward for a maths problem", cmd_verify),
    "dpo": ("2.3 DPO loss and the PPO KL penalty on toy log-probabilities", cmd_dpo),
    "metrics": ("2.4 EM, F1, BLEU, ROUGE-L and perplexity", cmd_metrics),
    "contamination": ("2.4 find eval questions copied into the web crawl", cmd_contamination),
    "arena": ("2.4 Elo and Bradley-Terry ratings from pairwise votes", cmd_arena),
    "judge": ("2.4 position and verbosity checks for an LLM judge", cmd_judge),
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab", description="Module 02 lab: decoding, post-training, evaluation."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, _) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        if name in ("next-token", "generate"):
            p.add_argument("--prompt", default="to reset the")
        if name == "generate":
            p.add_argument("--strategy", choices=["greedy", "beam", "sample"], default="sample")
            p.add_argument("--temperature", type=float, default=1.0)
            p.add_argument("--top-k", type=int, default=0)
            p.add_argument("--top-p", type=float, default=1.0)
            p.add_argument("--repetition-penalty", type=float, default=1.0)
            p.add_argument("--num-beams", type=int, default=4)
            p.add_argument("--max-tokens", type=int, default=40)
            p.add_argument("--stop", action="append", default=[], help="stop sequence (repeatable)")
            p.add_argument("--seed", type=int, default=None)
        if name == "chat-template":
            p.add_argument("--id", default="s002", help="row id in sft/instructions.jsonl")
    args = parser.parse_args(argv)
    COMMANDS[args.command][1](args)


if __name__ == "__main__":
    main()
