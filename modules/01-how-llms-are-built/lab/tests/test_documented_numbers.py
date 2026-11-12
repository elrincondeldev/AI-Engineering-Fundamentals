"""Every number quoted in the module's Markdown comes from here.

Two kinds of checks:

1. ``test_expected_output_blocks_match_real_runs`` re-runs every ``uv run lab ...`` command that is
   immediately followed by a ``text`` block in the Markdown, and checks each line of the block.
2. The other tests assert the numbers quoted in prose and tables, lesson by lesson, exactly or with
   the stated rounding.

If you change the lab, run this file and update the Markdown it points to.
"""

from __future__ import annotations

import contextlib
import io
import math
import re
import shlex
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from m01_foundations import bpe, cleaning, ngram, params, published, transformer
from m01_foundations.cli import _default_clean, default_tokenizer, main, ticket_bodies
from m01_foundations.data import read_jsonl
from m01_foundations.dedup import MinHasher, estimated_jaccard, jaccard, lsh_threshold, shingles
from m01_foundations.neural import count_bigram_loss, train_neural_bigram

MODULE = Path(__file__).resolve().parents[2]
FENCE = re.compile(r"^\s*```(\w*)\s*$")


@lru_cache(maxsize=None)
def run(command: str) -> str:
    """Run ``uv run lab <args>`` in-process and return what it prints."""
    args = shlex.split(command)[3:]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            main(args)
        except SystemExit:  # --help exits
            pass
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # `lab clean` writes outputs/ in the working directory
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("COLUMNS", "80")


def fenced_blocks(path: Path) -> list[tuple[str, list[str]]]:
    blocks, lang, body, inside = [], "", [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        m = FENCE.match(line)
        if m and not inside:
            inside, lang, body = True, m.group(1), []
        elif m and inside and not m.group(1):
            blocks.append((lang, body))
            inside = False
        elif inside:
            body.append(line)
    return blocks


def command_output_pairs() -> list[tuple[str, str, list[str]]]:
    pairs = []
    for md in sorted(MODULE.rglob("*.md")):
        if ".venv" in md.parts or md.name.endswith(".es.md"):
            continue
        blocks = fenced_blocks(md)
        for (lang_a, body_a), (lang_b, body_b) in zip(blocks, blocks[1:]):
            cmds = [ln.strip() for ln in body_a if ln.strip().startswith("uv run lab")]
            if lang_a == "bash" and lang_b == "text" and len(cmds) == 1:
                pairs.append((md.name, cmds[0], body_b))
    return pairs


PAIRS = command_output_pairs()


def test_markdown_has_expected_output_blocks() -> None:
    assert len(PAIRS) >= 12


@pytest.mark.parametrize(("page", "command", "expected"), PAIRS, ids=[f"{p}:{c}" for p, c, _ in PAIRS])
def test_expected_output_blocks_match_real_runs(page: str, command: str, expected: list[str]) -> None:
    printed = {line.strip() for line in run(command).splitlines()}
    for line in expected:
        if line.strip():
            assert line.strip() in printed, f"{page}: {line.strip()!r} not printed by {command!r}"


EXTRA_COMMANDS = [
    "uv run lab params --show classic-small",
    "uv run lab bpe --es-weight 5",
    "uv run lab clean --min-words 50",
    "uv run lab clean --skip near_dup",
]


def test_every_text_block_line_was_printed_by_some_command() -> None:
    """Catch-all: text blocks that are not right after their command must still be real output."""
    printed: set[str] = set()
    for command in [c for _, c, _ in PAIRS] + EXTRA_COMMANDS:
        printed |= {line.strip() for line in run(command).splitlines()}
    for md in sorted(MODULE.rglob("*.md")):
        if ".venv" in md.parts or md.name.endswith(".es.md"):
            continue
        for lang, body in fenced_blocks(md):
            lines = [ln.strip() for ln in body if ln.strip()]
            if lang != "text" or not lines or lines[0].startswith("raw crawl"):
                continue
            if re.match(r"^\d+ passed", lines[0]):
                continue
            for line in lines:
                assert line in printed, f"{md.name}: {line!r} is not output of any lab command"


def test_lab_readme_test_count(request: pytest.FixtureRequest) -> None:
    files = {Path(str(item.fspath)).name for item in request.session.items}
    if not {"test_providers.py", "test_bpe.py", "test_cleaning.py"} <= files:
        pytest.skip("only meaningful when the whole suite runs")
    readme = (MODULE / "lab" / "README.md").read_text(encoding="utf-8")
    documented = int(re.search(r"^(\d+) passed", readme, re.M).group(1))
    assert documented == request.session.testscollected


# --------------------------------------------------------------------------- #
# Lesson 1.1                                                                  #
# --------------------------------------------------------------------------- #


def _english_model() -> ngram.NGramModel:
    docs, _ = _default_clean()
    return ngram.NGramModel(order=4).fit(ngram.corpus_tokens(cleaning.kept_texts(docs, "en")))


def test_lesson_1_1_next_word_and_wrong_date() -> None:
    docs, _ = _default_clean()
    assert len(cleaning.kept_texts(docs, "en")) == 77
    model = _english_model()
    used, probs = model.next_word_probs(ngram.words("Lower the setpoint by one degree in the"))
    assert used == 3
    assert [(w, round(p, 3)) for w, p in probs] == [
        ("living", 0.364), ("bedroom", 0.273), ("bathroom", 0.091), ("hallway", 0.091), ("home", 0.091)]
    assert " ".join(model.generate("The Ember T3")) == "costs € 149 and launched on 2023 - 05 - 22 ."
    assert dict(model.counts[("on", "2023", "-")]) == {"05": 1, "10": 1}  # the tie
    text = " ".join(cleaning.kept_texts(docs))
    assert "The Ember T3 costs €149 and launched on 2023-10-02." in text
    assert "arrived in shops on 2023-05-22" in text


def test_lesson_1_1_coverage_table() -> None:
    docs, _ = _default_clean()
    train, held = ngram.split_docs(cleaning.kept_texts(docs, "en"))
    assert (len(train), len(held)) == (61, 16)
    tr, ho = ngram.corpus_tokens(train), ngram.corpus_tokens(held)
    got = [round(ngram.coverage(tr, ho, n), 3) for n in range(1, 6)]
    assert got == [0.973, 0.855, 0.735, 0.645, 0.576]


def test_lesson_1_1_cited_scale_figures() -> None:
    assert published.fact("GPT-3", "parameters") == 175
    assert published.fact("Chinchilla", "parameters") == 70
    assert published.fact("Chinchilla", "Gopher") == 280
    assert params.training_flops(1, 1) == 6


# --------------------------------------------------------------------------- #
# Lesson 1.2                                                                  #
# --------------------------------------------------------------------------- #


def test_lesson_1_2_crawl_and_pipeline_counts() -> None:
    docs, rep = _default_clean()
    assert rep.total == 134
    assert sum(d.status == 200 for d in docs) == 128
    assert sum(d.content_type == "text/html" for d in docs) == 131
    assert sum(r["keep"] for r in read_jsonl("web/labels.jsonl")) == 83
    assert rep.dropped_by_stage == {"fetch": 9, "url_dedup": 4, "url_filter": 0, "extract": 0, "language": 8,
                                    "quality": 13, "exact_dup": 8, "near_dup": 8, "contamination": 2}
    assert len(cleaning.STAGES) == 9  # "a nine-stage cleaning pipeline"
    assert rep.kept == 82 and rep.total - rep.kept == 52
    assert (rep.text_chars_kept, rep.html_chars) == (48_932, 192_610)
    assert round(rep.text_chars_kept / rep.html_chars, 3) == 0.254
    langs = {lang: sum(1 for d in docs if d.kept and d.lang == lang) for lang in ("en", "es")}
    assert langs == {"en": 77, "es": 5}
    reasons = [d.dropped for d in docs if not d.kept]
    assert reasons.count("language:de") == 4 and reasons.count("language:fr") == 4
    assert rep.pii_pages == 4 and rep.pii_counts == {"ADDRESS": 1, "EMAIL": 2, "PHONE": 2}
    assert (rep.tp, rep.fp, rep.fn, rep.tn) == (82, 0, 1, 51)
    assert (round(rep.precision, 3), round(rep.recall, 3), round(rep.f1, 3)) == (1.0, 0.988, 0.994)
    assert rep.false_drops == [("https://corvid-home.example/thermo-x", "quality:repetitive")]


def test_lesson_1_2_quality_details() -> None:
    docs, _ = _default_clean()
    labels = {r["url"]: r for r in read_jsonl("web/labels.jsonl")}
    quality = [d for d in docs if d.dropped.startswith("quality")]
    assert sum(labels[d.url]["is_spam"] for d in quality) == 9
    assert sum(d.dropped == "quality:too_short" for d in quality) == 3
    by_url = {d.url: d for d in docs}
    spam = cleaning.quality_signals(by_url["https://best-cheap-smart-deals.example/deal-1"].text)
    assert round(spam["top_trigram_share"], 2) == 0.59
    corvid = cleaning.quality_signals(by_url["https://corvid-home.example/thermo-x"].text)
    assert corvid["words"] == 19 and round(corvid["top_trigram_share"], 2) == 0.32
    kinds = [labels[u] for u in labels]
    assert sum("utm_" in r["url"] for r in kinds) == 4
    assert sum("/syndicated/" in r["url"] for r in kinds) == 8
    assert sum(bool(r["near_duplicate_of"]) for r in kinds) == 8
    assert sum(bool(r["contamination"]) for r in kinds) == 2


@pytest.mark.parametrize(
    ("flags", "kept", "precision", "recall", "f1"),
    [
        ("--min-words 50", 57, "1.000", "0.687", "0.814"),
        ("--skip contamination", 84, "0.976", "0.988", "0.982"),
        ("--skip near_dup", 90, "0.911", "0.988", "0.948"),
        ("--skip url_dedup --skip exact_dup --skip near_dup", 101, "0.812", "0.988", "0.891"),
    ],
)
def test_lesson_1_2_ablation_table(flags: str, kept: int, precision: str, recall: str, f1: str) -> None:
    out = run(f"uv run lab clean {flags}")
    assert f"Kept {kept} pages" in out
    assert f"precision={precision}  recall={recall}  F1={f1}" in out


def test_lesson_1_2_minhash_numbers() -> None:
    assert round(lsh_threshold(16, 8), 3) == 0.707
    docs, _ = cleaning.clean_crawl(cleaning.PipelineConfig(use_near_dedup=False))
    by_url = {d.url: d for d in docs}
    hasher = MinHasher()
    exact, est = [], []
    for row in read_jsonl("web/labels.jsonl"):
        if row["near_duplicate_of"]:
            a, b = shingles(by_url[row["url"]].text), shingles(by_url[row["near_duplicate_of"]].text)
            exact.append(round(jaccard(a, b), 3))
            est.append(round(estimated_jaccard(hasher.signature(a), hasher.signature(b)), 3))
    assert (min(exact), max(exact)) == (0.968, 0.975)
    assert (min(est), max(est)) == (0.953, 0.992)
    assert "Most similar pair of different articles: J=0.417" in run("uv run lab minhash")


def test_lesson_1_2_cited_dataset_sizes() -> None:
    assert published.fact("FineWeb", "tokens") == 15
    assert published.fact("FineWeb", "Common Crawl snapshots") == 96
    assert published.fact("RefinedWeb", "tokens extracted") == 5
    assert published.fact("RefinedWeb", "tokens in the public extract") == 600
    assert published.fact("Dolma", "tokens") == 3


# --------------------------------------------------------------------------- #
# Lesson 1.3                                                                  #
# --------------------------------------------------------------------------- #


def test_lesson_1_3_tokenizer_numbers() -> None:
    tok = default_tokenizer()
    en, es = "How long does shipping take?", "¿Cuánto tarda el envío?"
    assert (len(en.encode()), len(en.split()), len(tok.encode(en))) == (28, 5, 12)
    assert (len(es.split()), len(tok.encode(es))) == (4, 19)
    assert tok.vocab_size == 1024 and len(tok.merges) == 767
    assert tok.special == {bpe.END_OF_TEXT: 1023}
    assert tok.encode(f"Hola{bpe.END_OF_TEXT}Hello") == [72, 316, 97, 1023, 72, 744, 111]
    docs, _ = _default_clean()
    texts = cleaning.kept_texts(docs)
    assert len(texts) == 82 and sum(len(t.encode()) for t in texts) == 48_999
    assert len(cleaning.kept_texts(docs, "es")) == 5
    _, history = bpe.train_bpe(texts, 1024)
    first = [(h.piece, h.count) for h in history[:6]]
    assert first == [(b" t", 1003), (b"he", 955), (b" a", 900), (b" s", 797), (b"in", 672), (b" the", 657)]


def test_lesson_1_3_tokens_per_word_tables() -> None:
    tickets = ticket_bodies()
    assert (len(tickets["en"]), len(tickets["es"])) == (846, 378)
    assert round(len(tickets["es"]) / (846 + 378), 2) == 0.31  # "about 30% of tickets are Spanish"
    expected = {257: (5.566, 5.726, 1.029), 384: (3.864, 4.193, 1.085), 512: (3.519, 3.907, 1.110),
                768: (3.139, 3.565, 1.136), 1024: (2.932, 3.345, 1.141)}
    tok = default_tokenizer()
    for size, (e, s, ratio) in expected.items():
        t = tok.truncated(size - 257)
        en, es = bpe.tokens_per_word(t, tickets["en"]), bpe.tokens_per_word(t, tickets["es"])
        assert (round(en, 3), round(es, 3), round(es / en, 3)) == (e, s, ratio)
    weighted = default_tokenizer(1024, 5)
    en, es = bpe.tokens_per_word(weighted, tickets["en"]), bpe.tokens_per_word(weighted, tickets["es"])
    assert (round(en, 3), round(es, 3), round(es / en, 3)) == (2.993, 3.231, 1.080)


def test_lesson_1_3_tiny_corpus_and_cited_vocab() -> None:
    tiny = ["low lower lowest", "new newer newest", "the lowest price is the newest price"] * 3
    _, history = bpe.train_bpe(tiny, 267)
    assert (history[0].piece, history[0].count) == (b"we", 18)
    assert published.fact("Qwen2.5", "regular tokens") == 151_643


# --------------------------------------------------------------------------- #
# Lesson 1.4                                                                  #
# --------------------------------------------------------------------------- #


def _train_tokens() -> tuple[np.ndarray, np.ndarray]:
    tok = default_tokenizer()
    docs, _ = _default_clean()
    tr, va = ngram.split_docs(cleaning.kept_texts(docs))
    enc = lambda ts: np.array([i for t in ts for i in tok.encode(t + bpe.END_OF_TEXT)])  # noqa: E731
    return enc(tr), enc(va)


def test_lesson_1_4_training_curves() -> None:
    train, valid = _train_tokens()
    assert (len(train), len(valid)) == (11_204, 2_596)
    res = train_neural_bigram(train, valid, 1024, steps=1000, lr=10.0, eval_every=200)
    got = [(s, round(a, 3), round(b, 3)) for s, a, b in res.history]
    assert got == [(0, 6.932, 6.932), (200, 4.237, 4.6), (400, 2.832, 3.376), (600, 2.297, 3.036),
                   (800, 2.042, 2.986), (1000, 1.914, 2.985)]
    assert round(res.uniform_loss, 3) == 6.931 == round(math.log(1024), 3)
    assert round(count_bigram_loss(train, valid, 1024), 3) == 3.065
    slow = train_neural_bigram(train, valid, 1024, steps=1000, lr=0.5, eval_every=200)
    assert round(slow.history[-1][2], 3) == 5.887
    with np.errstate(all="ignore"):
        boom = train_neural_bigram(train, valid, 1024, steps=200, lr=100.0, eval_every=200)
    assert math.isnan(boom.history[-1][2])


def test_lesson_1_4_attention_numbers() -> None:
    cfg = transformer.TinyConfig()
    shape = (cfg.vocab_size, cfg.d_model, cfg.n_heads, cfg.d_head, cfg.n_layers, cfg.d_ff)
    assert shape == (1024, 32, 4, 8, 2, 128)
    tok = default_tokenizer()
    ids = np.array(tok.encode("Lower the setpoint by one degree"))
    assert len(ids) == 7
    logits, weights = transformer.forward(ids, transformer.init_params(cfg), cfg)
    assert logits.shape == (7, 1024) and weights[0].shape == (4, 7, 7)
    assert round(transformer.cross_entropy(logits[:-1], ids[1:]), 3) == 6.905
    assert params.count_params(params.LAB_TINY).total == 57_984


def test_lesson_1_4_ffn_is_twice_attention_in_classic_small() -> None:
    b = params.count_params(params.CLASSIC_SMALL).breakdown
    assert round(b["feed_forward"] / b["attention"], 1) == 2.0


# --------------------------------------------------------------------------- #
# Lesson 1.5                                                                  #
# --------------------------------------------------------------------------- #


def test_lesson_1_5_classic_small() -> None:
    c = params.count_params(params.CLASSIC_SMALL)
    assert c.total == 124_439_808
    shares = {k: round(v / c.total * 100, 1) for k, v in c.breakdown.items()}
    assert (shares["embeddings"], shares["positions"], shares["attention"], shares["feed_forward"]) == (
        31.0, 0.6, 22.8, 45.5)
    assert c.non_embedding == 85_056_000
    assert params.kaplan_estimate(params.CLASSIC_SMALL) == 84_934_656
    assert (c.non_embedding - 84_934_656) / c.non_embedding < 0.002
    memory = [round(params.weight_memory_gb(c.total, d), 2) for d in ("fp32", "bf16", "int4")]
    assert memory == [0.5, 0.25, 0.06]
    assert f"{params.training_flops(c.active, 1e12):.2e}" == "7.47e+20"


def test_lesson_1_5_moe_and_dense_presets() -> None:
    dense = params.count_params(params.MODERN_1B)
    moe = params.count_params(params.MOE_8X)
    fine = params.count_params(params.MOE_FINE)
    assert (dense.total, moe.total, moe.active, fine.total, fine.active) == (
        852_559_872, 4_728_358_912, 1_406_470_144, 4_868_605_952, 854_657_024)
    assert moe.total / dense.total > 5
    assert round(fine.total / dense.total, 1) == 5.7  # "almost six times"
    assert round(moe.breakdown["feed_forward"] / moe.total * 100, 1) == 93.7
    assert round(params.weight_memory_gb(moe.total), 2) == 9.46
    assert round(dense.total / params.count_params(params.CLASSIC_SMALL).total, 1) == 6.9  # "about seven"


def test_lesson_1_5_kv_cache() -> None:
    assert round(params.kv_cache_bytes(params.MODERN_1B, 1000) / 1e6, 2) == 32.77
    assert round(params.kv_cache_bytes(params.CLASSIC_SMALL, 1000) / 1e6, 2) == 36.86
    m = params.MODERN_1B
    assert (m.n_heads, m.kv_heads, m.context_length) == (16, 4, 8192)
    full = params.kv_cache_bytes(params.MODERN_1B, 8192)
    local = params.kv_cache_bytes(params.MODERN_1B, 8192, local_window=1024, local_per_global=5)
    assert (round(full / 1e9, 3), round(local / 1e9, 3)) == (0.268, 0.063)


def test_lesson_1_5_cited_figures_and_shares() -> None:
    f = published.fact
    ds_total, ds_active = f("DeepSeek-V3", "total"), f("DeepSeek-V3", "parameters activated")
    assert round(params.active_share(ds_total, ds_active), 3) == 0.055
    assert round(f("DeepSeek-V3", "total") / f("DeepSeek-V3", "parameters activated")) == 18
    experts = (f("DeepSeek-V3", "routed experts per"), f("DeepSeek-V3", "shared"),
               f("DeepSeek-V3", "routed experts act"))
    assert experts == (256, 1, 8)
    assert f("DeepSeek-V3", "pre-training tokens") == 14.8
    mx_total, mx_active = f("Mixtral 8x7B", "parameters each"), f("Mixtral 8x7B", "active")
    assert round(params.active_share(mx_total, mx_active), 3) == 0.277
    assert (f("Mixtral 8x7B", "experts per layer"), f("Mixtral 8x7B", "experts per token")) == (8, 2)
    assert (f("Qwen2.5", "smallest"), f("Qwen2.5", "largest"), f("Qwen2.5", "pre-training")) == (0.5, 72, 18)
    assert (f("Gemma 3", "smallest"), f("Gemma 3", "largest"), f("Gemma 3", "context")) == (1, 27, 128)
    assert (f("Gemma 3", "local attention layers"), f("Gemma 3", "local attention window")) == (5, 1024)


# --------------------------------------------------------------------------- #
# Quiz, lab README Break-it and Extend-it                                     #
# --------------------------------------------------------------------------- #


def test_quiz_calculations() -> None:
    _, history = bpe.train_bpe(["ab ab ab ba"], 258)
    assert (history[0].piece, history[0].count) == (b"ab", 3)
    small = params.ModelShape("q10", vocab_size=1000, d_model=512, n_layers=8, n_heads=8, d_ff=2048,
                              context_length=512)
    assert params.kaplan_estimate(small) == 25_165_824
    assert (200 + 8 * 100, 200 + 2 * 100) == (1000, 400)
    full = params.kv_cache_bytes(params.MODERN_1B, 8192)
    no_gqa = params.kv_cache_bytes(replace(params.MODERN_1B, n_kv_heads=None), 8192)
    assert no_gqa == 4 * full


def test_break_it_no_utm_stripping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cleaning, "TRACKING_PARAMS", ("fbclid",))
    _, rep = cleaning.clean_crawl()
    assert rep.kept == 82
    stages = rep.dropped_by_stage
    assert (stages["url_dedup"], stages["exact_dup"], stages["quality"]) == (0, 11, 14)


def test_break_it_low_near_dup_threshold_changes_nothing() -> None:
    _, rep = cleaning.clean_crawl(cleaning.PipelineConfig(near_dup_threshold=0.3))
    assert rep.kept == 82


def test_break_it_wrong_merge_order(monkeypatch: pytest.MonkeyPatch) -> None:
    tok = default_tokenizer()
    tickets = ticket_bodies()["en"]

    def by_latest(self: bpe.BPETokenizer, chunk: str) -> tuple[int, ...]:
        ids = tuple(chunk.encode("utf-8"))
        while len(ids) >= 2:
            ranked = [(self.merges[p], p) for p in zip(ids, ids[1:]) if p in self.merges]
            if not ranked:
                break
            new_id, pair = max(ranked)
            ids = bpe.merge_pair(ids, pair, new_id)
        return ids

    monkeypatch.setattr(bpe.BPETokenizer, "_encode_chunk", by_latest)
    fresh = bpe.BPETokenizer(merges=tok.merges, special=tok.special, vocab=tok.vocab)
    assert round(bpe.tokens_per_word(fresh, tickets), 3) == 3.360
    assert all(fresh.decode(fresh.encode(t)) == t for t in tickets[:50])


def test_extend_it_repetition_rule_needs_30_words(monkeypatch: pytest.MonkeyPatch) -> None:
    original = cleaning.quality_failures

    def guarded(text: str, cfg: cleaning.QualityConfig = cleaning.QualityConfig()) -> list[str]:
        fails = original(text, cfg)
        if "repetitive" in fails and cleaning.quality_signals(text)["words"] < 30:
            fails.remove("repetitive")
        return fails

    monkeypatch.setattr(cleaning, "quality_failures", guarded)
    docs, rep = cleaning.clean_crawl()
    assert rep.kept == 83 and rep.precision == 1.0 and rep.recall == 1.0
    labels = {r["url"]: r for r in read_jsonl("web/labels.jsonl")}
    spam = [d for d in docs if labels[d.url]["is_spam"] and d.text]
    spam_words = [cleaning.quality_signals(d.text)["words"] for d in spam]
    assert min(spam_words) == 45


def test_extend_it_pii_report() -> None:
    docs, _ = _default_clean()
    labels = {r["url"]: r for r in read_jsonl("web/labels.jsonl")}
    labelled = [d for d in docs if d.kept and labels[d.url]["has_pii"]]
    flagged = [d for d in docs if d.kept and d.pii]
    assert len(labelled) == 4 and {d.url for d in flagged} == {d.url for d in labelled}
