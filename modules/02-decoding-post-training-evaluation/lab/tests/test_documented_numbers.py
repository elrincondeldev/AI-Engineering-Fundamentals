"""Every number quoted in the module's Markdown comes from a real run of this lab.

Two kinds of checks:

1. ``test_expected_output_blocks``: every ```bash block with ``uv run lab ...`` commands that is
   followed by a ```text block (in the lab README and the lessons) is executed, and the output must
   match the text block exactly.
2. The other tests assert the numbers that appear in prose and tables.
"""

from __future__ import annotations

import io
import math
import re
import shlex
from contextlib import redirect_stdout
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from m02_post_training import cli, decoding, evaluation, reward, sft
from m02_post_training.data import iter_csv, read_jsonl
from m02_post_training.ngram import EOS, NgramLM
from m02_post_training.text import load_corpus, tokenize

MODULE = Path(__file__).resolve().parents[2]
DOCS = [
    MODULE / "lab" / "README.md",
    MODULE / "01-decoding-strategies.md",
    MODULE / "02-supervised-fine-tuning.md",
    MODULE / "03-rlhf-and-preference-optimization.md",
    MODULE / "04-evaluating-llms.md",
]
FENCE = re.compile(r"^(\s*)```(\w*)\s*$")


def fenced_blocks(path: Path) -> list[tuple[str, str]]:
    """(language, dedented body) of every fenced block, in order."""
    blocks, lang, indent, body, inside = [], "", "", [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        m = FENCE.match(line)
        if not inside and m:
            inside, indent, lang, body = True, m.group(1), m.group(2), []
        elif inside and m and m.group(2) == "":
            blocks.append((lang, "\n".join(body)))
            inside = False
        elif inside:
            body.append(line[len(indent):] if line.startswith(indent) else line)
    return blocks


def command_output_pairs() -> list[tuple[str, list[list[str]], str]]:
    pairs = []
    for path in DOCS:
        blocks = fenced_blocks(path)
        for (lang, body), (next_lang, expected) in zip(blocks, blocks[1:]):
            if lang != "bash" or next_lang != "text":
                continue
            cmds = [shlex.split(ln)[3:] for ln in body.splitlines() if ln.startswith("uv run lab ")]
            if cmds:
                pairs.append((f"{path.name}:{body.splitlines()[0]}", cmds, expected))
    return pairs


@cache
def run_lab(*argv: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main(list(argv))
    return buf.getvalue()


def normalise(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


PAIRS = command_output_pairs()


def test_docs_contain_expected_output_blocks() -> None:
    assert len(PAIRS) >= 20


@pytest.mark.parametrize(("where", "cmds", "expected"), PAIRS, ids=[p[0] for p in PAIRS])
def test_expected_output_blocks(where: str, cmds: list[list[str]], expected: str) -> None:
    actual = "".join(run_lab(*c) for c in cmds)
    assert normalise(actual) == normalise(expected), where


# --------------------------------------------------------------------------- #
# Lesson 2.1                                                                  #
# --------------------------------------------------------------------------- #


@cache
def lm() -> NgramLM:
    return cli.build_lm()


def test_corpus_numbers() -> None:
    docs = load_corpus()
    assert len(docs) == 95
    assert sum(not d.source.startswith("http") for d in docs) == 17
    assert lm().n_train_tokens == 12_544 and len(lm().vocab) == 797
    assert lm().lambdas == (0.1, 0.3, 0.6)  # 0.1 word frequency, 0.3 bigram, 0.6 trigram


def test_temperature_table_and_nucleus() -> None:
    p = {t: dict(decoding.top_tokens(lm(), "to reset the", t))["device"] for t in (0.5, 1.0, 2.0)}
    assert (round(p[0.5], 3), round(p[1.0], 3), round(p[2.0], 3)) == (0.996, 0.612, 0.063)
    assert round(0.9 - p[1.0], 2) == 0.29
    assert decoding.nucleus_size(lm(), "to reset the", 0.5) == 1
    assert decoding.nucleus_size(lm(), "to reset the", 0.9) == 79
    assert decoding.nucleus_size(lm(), "if your bulb", 0.5) == 8
    assert decoding.nucleus_size(lm(), "if your bulb", 0.9) == 118


def test_beam_search_diagram_scores() -> None:
    ctx = tokenize("to reset the")

    def score(words: str) -> float:
        total, history = 0.0, list(ctx)
        for w in words.split():
            total += float(decoding.next_logits(lm(), history)[lm().index[w]])
            history.append(w)
        return round(total, 2)

    assert score("device") == -0.49 and score("router") == -4.47
    assert score("device and") == -2.08 and score("device keeps") == -2.43
    assert score("device and choose") == -2.84 and score("device keeps working") == -2.88
    beam = decoding.generate(lm(), "to reset the", decoding.DecodeConfig(strategy="beam", num_beams=4))
    assert beam.text == "device keeps working during the holidays." and round(beam.logprob, 2) == -7.31
    greedy = decoding.generate(lm(), "to reset the", decoding.DecodeConfig(strategy="greedy"))
    assert len(greedy.tokens) == 37 and round(greedy.logprob, 2) == -29.59


def test_loops_penalties_and_seeds() -> None:
    cfg = decoding.DecodeConfig(strategy="greedy", max_new_tokens=50)
    loop = decoding.generate(lm(), "returns are", cfg)
    penalised = decoding.DecodeConfig(**{**cfg.__dict__, "repetition_penalty": 1.3})
    fixed = decoding.generate(lm(), "returns are", penalised)
    assert round(decoding.repetition_rate(loop.tokens), 2) == 0.38
    assert round(decoding.repetition_rate(fixed.tokens), 2) == 0.0
    hot = decoding.generate(lm(), "to reset the", decoding.DecodeConfig(temperature=1.5, seed=7))
    assert round(hot.logprob) == -247
    s7 = decoding.generate(lm(), "to reset the", decoding.DecodeConfig(top_p=0.9, seed=7))
    s7b = decoding.generate(lm(), "to reset the", decoding.DecodeConfig(top_p=0.9, seed=7))
    s8 = decoding.generate(lm(), "to reset the", decoding.DecodeConfig(top_p=0.9, seed=8))
    assert s7.text == s7b.text != s8.text


def test_eos_appears_once_per_document() -> None:
    docs = [tokenize(d.text) for d in load_corpus()]
    assert len(docs) == 95 and sum(len(d) + 1 for d in docs) == 12_544  # +1 = one <eos> per document
    assert EOS in lm().index


def test_decoding_benchmark_table() -> None:
    rows = {r.name: r for r in decoding.decoding_benchmark(lm())}
    expected = {
        "greedy": (0.130, 0.062, 6, -0.78, 0.83),
        "beam (4 beams)": (0.140, 0.000, 6, -0.85, 1.00),
        "sample T=1.0": (0.540, 0.021, 30, -2.00, 0.03),
        "sample T=0.7": (0.532, 0.006, 30, -1.12, 0.40),
        "sample T=1.5": (0.623, 0.000, 30, -5.77, 0.40),
        "top-k k=10": (0.616, 0.008, 30, -1.26, 0.27),
        "top-p p=0.9": (0.635, 0.001, 30, -1.38, 0.37),
        "top-p 0.9 + rep 1.3": (0.655, 0.003, 30, -1.32, 0.23),
    }
    for name, (d2, rep3, uniq, lp, eos) in expected.items():
        r = rows[name]
        assert (round(r.distinct_2, 3), round(r.rep_3, 3), r.unique_outputs) == (d2, rep3, uniq), name
        assert (round(r.logprob_per_token, 2), round(r.eos_rate, 2), r.n_outputs) == (lp, eos, 30), name
    assert len(decoding.BENCH_PROMPTS) == 6


# --------------------------------------------------------------------------- #
# Lesson 2.2                                                                  #
# --------------------------------------------------------------------------- #


def test_sft_numbers() -> None:
    rows, labels = sft.load_sft()
    report = sft.clean(rows)
    assert (len(rows), len(report.kept), len(report.dropped)) == (261, 108, 153)
    assert report.count("duplicate") == 132
    assert len(report.dropped) - report.count("duplicate") == 21
    assert sft.duplicates_resolved(report, rows, labels) == (6, 6)
    assert len(sft.question_leaks(report.kept)) == 5
    assert len(sft.response_repeats(report.kept)) == 25
    before, after = sft.mask_stats(rows), sft.mask_stats(report.kept)
    assert (before.total_tokens, round(before.trained_share, 2)) == (13_874, 0.42)
    assert (after.total_tokens, round(after.trained_share, 2)) == (5_969, 0.42)
    scores = {s.flag: (s.labelled, s.flagged, s.correct, round(s.precision, 2), s.recall)
              for s in sft.score_against_labels(report, labels)}
    assert scores["contaminated_rag_eval"] == (5, 11, 5, 0.45, 1.0)
    assert scores["outdated_policy"] == (4, 4, 4, 1.0, 1.0)
    assert scores["pii_in_response"] == (3, 3, 3, 1.0, 1.0)
    for flag in ("empty_response", "low_quality", "wrong_language"):
        assert scores[flag] == (1, 1, 1, 1.0, 1.0)


def test_unlabelled_contamination_hits_copy_q027() -> None:
    rows, labels = sft.load_sft()
    report = sft.clean(rows)
    index = sft.ContaminationIndex.from_rag_eval()
    assert index.n == 8
    flagged = [r for r in rows if report.dropped.get(r["id"]) == "contaminated_rag_eval"]
    extra = [r for r in flagged if not labels[r["id"]]]
    assert len(extra) == 6
    assert all(index.matches(f"{r['instruction']} {r['response']}") == ["q027"] for r in extra)
    q027 = next(q for q in read_jsonl("rag_eval/qa.jsonl") if q["id"] == "q027")
    assert all(r["response"] == q027["reference_answer"] for r in extra)


def test_mask_of_s002() -> None:
    rows, _ = sft.load_sft()
    row = next(r for r in rows if r["id"] == "s002")
    toks = sft.chat_tokens(sft.render_chat(sft.row_to_messages(row)))
    assert (len(toks), sum(sft.loss_mask(toks))) == (42, 11)
    assert round(11 / 42, 2) == 0.26


def test_story_facts_quoted_from_the_data() -> None:
    ticket = next(t for t in iter_csv("tickets/tickets.csv") if t["ticket_id"] == "T00192")
    assert "¿Cómo devuelvo un Spark Mini que compré hace 14 días?" in ticket["body"]
    q033 = next(q for q in read_jsonl("rag_eval/qa.jsonl") if q["id"] == "q033")
    assert "14 days" in q033["reference_answer"]
    assert len(read_jsonl("rag_eval/qa.jsonl")) == 40
    assert len(read_jsonl("reasoning/problems.jsonl")) == 160


# --------------------------------------------------------------------------- #
# Lesson 2.3                                                                  #
# --------------------------------------------------------------------------- #


def test_preference_data_and_splits() -> None:
    train, test = reward.load_pairs("train"), reward.load_pairs("test")
    assert (len(train), len(test)) == (240, 60)
    prompts = {p["prompt"] for p in train}
    assert len(prompts) == 23 and {p["prompt"] for p in test} <= prompts
    train_a, test_b = reward.prompt_disjoint(train, test)
    assert (len(train_a), len(test_b), len(reward.length_balanced(train_a))) == (125, 28, 54)
    assert round(100 / 28, 1) == 3.6
    assert round(reward.chosen_is_longer(train), 3) == 0.787
    assert round(reward.chosen_is_longer(test), 3) == 0.233


def test_reward_tables() -> None:
    tables = reward.reward_experiments()
    got = {k: [(r.name, r.n_train_pairs, r.n_features, round(r.train_acc, 3), round(r.test_acc, 3),
                round(r.hack_rate, 3)) for r in rows] for k, rows in tables.items()}
    assert got["official"] == [
        ("length only", 240, 1, 0.787, 0.233, 0.767),
        ("words + length", 240, 302, 1.0, 1.0, 0.0),
    ]
    assert got["new_prompts"] == [
        ("length only", 125, 1, 0.784, 0.25, 0.75),
        ("words + length", 125, 217, 1.0, 0.714, 0.357),
        ("words + length, balanced", 54, 217, 1.0, 0.821, 0.25),
    ]


def test_ppo_and_dpo_numbers() -> None:
    assert reward.kl_penalized_reward(2.0, -12.0, -15.0, 0.1) == pytest.approx(1.70)
    assert 0.1 * (-12.0 - -15.0) == pytest.approx(0.3)
    table = {name: (margin, [round(x, 3) for x in losses]) for name, margin, losses in reward.dpo_table()}
    assert table["policy = reference (start of training)"] == (0.0, [0.693, 0.693])
    assert table["chosen up 1, rejected down 1"] == (2.0, [0.598, 0.313])
    assert table["chosen down 2, rejected down 5"] == (3.0, [0.554, 0.201])
    assert table["chosen down 1, rejected up 1 (wrong way)"] == (-2.0, [0.798, 1.313])
    assert round(math.log(2), 3) == 0.693


def test_quiz_calculations() -> None:
    assert round(float(reward.sigmoid(1.2 - 0.2)), 3) == 0.731
    assert reward.kl_penalized_reward(1.5, -10.0, -14.0, 0.2) == pytest.approx(0.7)
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    assert np.isfinite(decoding.top_p_filter(logits, 0.8)).sum() == 2
    assert np.isfinite(decoding.top_p_filter(logits, 0.81)).sum() == 3
    ratings = evaluation.elo_ratings([evaluation.Vote("a", "b", "a")])
    assert (round(ratings["a"]), round(ratings["b"])) == (1016, 984)


# --------------------------------------------------------------------------- #
# Lesson 2.4                                                                  #
# --------------------------------------------------------------------------- #


def test_metric_table() -> None:
    ref, rows = evaluation.metric_table()
    assert ref == "30 days from delivery."
    got = [(r.correct, round(r.em, 2), round(r.f1, 2), round(r.bleu, 2), round(r.rouge_l, 2)) for r in rows]
    assert got == [
        (True, 1.0, 1.0, 1.0, 1.0),
        (True, 0.0, 0.46, 0.18, 0.53),
        (True, 0.0, 0.36, 0.14, 0.40),
        (False, 0.0, 0.75, 0.75, 0.80),
        (True, 0.0, 0.33, 0.14, 0.34),
    ]


def test_perplexity_table() -> None:
    report = evaluation.perplexity_report()
    got = [(p.text, p.tokens, round(p.unk_share * 100), round(p.perplexity, 2)) for p in report]
    assert got == [
        ("training documents", 11_043, 4, 2.54),
        ("held-out documents", 1_406, 6, 7.01),
        ("greedy outputs", 219, 0, 2.66),
        ("top-p 0.9 outputs", 128, 0, 9.80),
        ("T=1.5 outputs", 97, 0, 674.91),
        ("Spanish help-centre pages", 266, 80, 20.30),
    ]


def test_contamination_table() -> None:
    rows, bad_pages, truth = evaluation.contamination_sweep()
    assert (bad_pages, truth) == (2, 12)
    got = [(r.n, r.min_share, r.pages_flagged, r.pages_correct, r.items_flagged, r.items_correct)
           for r in rows]
    assert got == [
        (3, 0.5, 2, 2, 46, 12),
        (8, 0.5, 2, 2, 45, 12),
        (8, 1.0, 2, 2, 12, 12),
        (13, 1.0, 2, 2, 12, 12),
    ]
    hits = evaluation.verbatim_hits(evaluation.load_web_texts(), evaluation.load_eval_items())
    assert sum(i.startswith("q") for _, i in hits) == 6 and sum(i.startswith("r") for _, i in hits) == 6


def test_arena_table() -> None:
    votes = evaluation.simulate_votes()
    assert len(votes) == 600 and len(evaluation.ARENA_STRENGTHS) == 4
    truth = {m: round(v) for m, v in evaluation.true_elo_scale().items()}
    elo = {m: round(v) for m, v in evaluation.elo_ratings(votes).items()}
    rev = {m: round(v) for m, v in evaluation.elo_ratings(votes[::-1]).items()}
    bt = {m: round(v) for m, v in evaluation.bradley_terry_ratings(votes).items()}
    bt_rev = {m: round(v) for m, v in evaluation.bradley_terry_ratings(votes[::-1]).items()}
    ci = {m: (round(a), round(b)) for m, (a, b) in evaluation.bootstrap_bt(votes).items()}
    assert truth == {"model-A": 1113, "model-B": 1026, "model-C": 991, "model-D": 870}
    assert elo == {"model-A": 1105, "model-B": 1021, "model-C": 986, "model-D": 887}
    assert rev == {"model-A": 1087, "model-B": 1078, "model-C": 985, "model-D": 850}
    assert bt == bt_rev == {"model-A": 1105, "model-B": 1026, "model-C": 993, "model-D": 875}
    assert ci == {
        "model-A": (1075, 1146),
        "model-B": (997, 1061),
        "model-C": (959, 1022),
        "model-D": (844, 903),
    }


def test_more_votes_separate_b_and_c() -> None:
    ci = evaluation.bootstrap_bt(evaluation.simulate_votes(n_votes=2400))
    assert ci["model-C"][1] < ci["model-B"][0]


def test_judge_table() -> None:
    test = reward.load_pairs("test")
    judge = evaluation.FakeProvider(responder=evaluation.biased_judge_responder())
    raw = evaluation.judge_swap_test(test, judge)
    stripped = evaluation.judge_swap_test(evaluation.strip_padding(test), judge)

    def row(r: evaluation.JudgeReport) -> tuple:
        return (r.n_pairs, *(round(x, 3) for x in (r.agreement_ab, r.agreement_ba, r.consistency,
                                                   r.first_position_rate, r.longer_rate)))

    assert row(raw) == (60, 0.233, 0.233, 1.0, 0.5, 1.0)
    assert row(stripped) == (60, 1.0, 0.667, 0.667, 0.667, 0.833)
    assert 60 * 2 * 2 == 240  # requests sent by `lab judge` (lab README, "Using a real model")
