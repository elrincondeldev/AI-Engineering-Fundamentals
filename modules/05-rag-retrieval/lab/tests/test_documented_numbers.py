"""Every number quoted in the module's Markdown comes from a real run and is asserted here.

Retrieval numbers on the knowledge base are exact (the pipeline is deterministic).
ANN numbers use small tolerances because FAISS and BLAS round floats slightly
differently on different CPUs. Timings are never asserted.
"""

from __future__ import annotations

import re
import shlex
import statistics
from pathlib import Path

import pytest

from m05_retrieval import ann as ann_module
from m05_retrieval.ann import AnnResult, make_vectors, run_benchmark
from m05_retrieval.bm25 import BM25, idf
from m05_retrieval.chunking import near_duplicates, prepare_corpus
from m05_retrieval.cli import LINEUP, main, run_lineup
from m05_retrieval.evaluate import EvalReport, evaluate, load_questions
from m05_retrieval.knowledge import KnowledgeIndex, extract_triples
from m05_retrieval.parsing import load_kb
from m05_retrieval.providers import count_tokens
from m05_retrieval.retrieval import (
    BM25Retriever,
    Config,
    HybridRetriever,
    RerankedRetriever,
    VectorRetriever,
    build_pipeline,
)
from m05_retrieval.text import tokenize, word_count
from m05_retrieval.vectors import HashingEmbedder, VectorIndex

# --------------------------------------------------------------------------- #
# Retrieval lineup (`uv run lab eval`)                                        #
# --------------------------------------------------------------------------- #

# name: (chunks, R@1, R@5, MRR, ctx@5, table R@5, cross-lingual R@5, exposed)
LINEUP_TABLE = {
    "bm25/heading/nofilter": (119, 0.611, 0.917, 0.745, 0.917, 0.00, 0.50, 15),
    "bm25/heading": (111, 0.667, 0.889, 0.770, 0.861, 0.00, 0.50, 0),
    "bm25/fixed": (65, 0.750, 0.972, 0.840, 0.944, 1.00, 0.50, 0),
    "bm25/recursive": (52, 0.778, 0.972, 0.863, 0.972, 1.00, 0.50, 0),
    "bm25/semantic": (98, 0.639, 0.917, 0.768, 0.917, 0.00, 0.50, 0),
    "bm25/table": (128, 0.667, 0.944, 0.786, 0.917, 1.00, 0.50, 0),
    "vector/table": (128, 0.722, 0.861, 0.804, 0.861, 0.00, 0.00, 0),
    "hybrid/table": (128, 0.750, 0.917, 0.816, 0.889, 0.50, 0.00, 0),
    "bm25/table/fold": (128, 0.667, 0.972, 0.800, 0.944, 1.00, 1.00, 0),
    "bm25/table/rerank": (128, 0.833, 0.972, 0.898, 0.944, 1.00, 0.50, 0),
    "bm25/table/fold/rerank": (128, 0.833, 1.000, 0.905, 0.972, 1.00, 1.00, 0),
}


@pytest.fixture(scope="module")
def lineup() -> dict[str, tuple[int, EvalReport]]:
    return {config.name: (chunks, report) for config, chunks, report in run_lineup()}


def test_lineup_table(lineup: dict[str, tuple[int, EvalReport]]) -> None:
    assert [c.name for c in LINEUP] == list(LINEUP_TABLE)
    for name, expected in LINEUP_TABLE.items():
        chunks, r = lineup[name]
        got = (chunks, r.recall_at_1, r.recall_at_5, r.mrr, r.context_at_5, r.by_category["table"],
               r.by_category["cross_lingual"], r.blocked_in_top5)
        assert got == expected, name


def test_misses_named_in_the_lessons(lineup: dict[str, tuple[int, EvalReport]]) -> None:
    assert lineup["bm25/heading"][1].misses == ["q004", "q017", "q018", "q022"]
    assert lineup["bm25/heading/nofilter"][1].misses == ["q017", "q018", "q022"]
    assert lineup["bm25/table"][1].misses == ["q004", "q022"]
    assert lineup["bm25/table/fold/rerank"][1].misses == []
    assert lineup["bm25/table"][1].by_category["policy"] == 0.90


def test_question_counts() -> None:
    assert len(load_questions()) == 36
    assert len(load_questions(answerable_only=False)) == 40
    # each answerable question is worth 1/36 of recall: about 0.028
    assert round(1 / 36, 3) == 0.028


# --------------------------------------------------------------------------- #
# Parsing, filtering, chunking                                                #
# --------------------------------------------------------------------------- #


def test_parse_numbers() -> None:
    docs = load_kb()
    tables = [b for d in docs for b in d.blocks if b.kind == "table"]
    assert (len(docs), len(tables), sum(len(t.rows) - 1 for t in tables)) == (26, 13, 80)
    assert sum(d.hidden_dropped for d in docs) == 1
    assert sum(count_tokens(d.text) for d in docs) == 5070


def test_filter_and_duplicate_numbers() -> None:
    docs = load_kb()
    kept, report = prepare_corpus(docs)
    assert len(kept) == 23 and len(report.dropped) == 3
    assert near_duplicates(docs) == [("kb-026", "kb-027", 0.884)]


@pytest.mark.parametrize(
    ("chunker", "count", "mean", "largest"),
    [("fixed", 65, 72.1, 94), ("recursive", 52, 73.5, 100), ("heading", 111, 34.4, 153),
     ("semantic", 98, 39.0, 120), ("table", 128, 31.1, 96)],
)
def test_chunk_sizes(chunker: str, count: int, mean: float, largest: int) -> None:
    sizes = [word_count(c.text) for c in build_pipeline(Config("bm25", chunker)).chunks]
    assert (len(sizes), round(statistics.mean(sizes), 1), max(sizes)) == (count, mean, largest)


def test_return_window_search_without_and_with_filter() -> None:
    q = "What is the return window for Hearthwise products?"
    raw = build_pipeline(Config("bm25", "heading", filter_meta=False)).retriever.search(q, 2)
    assert [(h.chunk.chunk_id, round(h.score, 3)) for h in raw] == [("kb-010#0", 11.765), ("kb-011#0", 6.756)]
    clean = build_pipeline(Config("bm25", "heading")).retriever.search(q, 1)
    assert (clean[0].chunk.chunk_id, round(clean[0].score, 3)) == ("kb-011#0", 6.997)


# --------------------------------------------------------------------------- #
# Indexes (`uv run lab index`) and the BM25 worked example                    #
# --------------------------------------------------------------------------- #


def test_index_sizes() -> None:
    chunks = build_pipeline(Config("bm25", "table")).chunks
    bm = BM25Retriever(chunks)
    idx = bm.bm25.index
    assert len(chunks) == 128
    assert sum(len(c.text.encode("utf-8")) for c in chunks) == 22048
    assert (len(idx.postings), idx.n_postings) == (950, 3361)
    assert idx.n_postings * 8 + sum(len(t.encode("utf-8")) for t in idx.postings) == 32064
    assert VectorRetriever(chunks).index.nbytes == 131072


def test_bm25_worked_example() -> None:
    pipe = build_pipeline(Config("bm25", "heading"))
    bm = pipe.retriever.bm25
    ids = [c.chunk_id for c in pipe.chunks]
    n = bm.index.n_docs
    assert (n, round(bm.avg_len, 1)) == (111, 34.4)
    df = {t: len(bm.index.postings[t]) for t in ("the", "watcher", "hearthwise")}
    assert df == {"the": 75, "watcher": 4, "hearthwise": 29}
    assert (round(idf(n, 75), 2), round(idf(n, 4), 2)) == (0.39, 3.21)
    assert "weigh" not in bm.index.postings
    table, reset = ids.index("kb-007#0"), ids.index("kb-021#4")
    assert (bm.index.lengths[table], bm.index.lengths[reset]) == (153, 17)
    watcher_idf = idf(n, df["watcher"])
    assert round(bm.term_score(1, 153, watcher_idf), 2) == 1.26
    assert round(bm.term_score(1, 17, watcher_idf), 2) == 4.16
    scores = bm.scores("How much does the Watcher S1 weigh?")
    ranking = sorted(range(n), key=lambda i: -scores[i])
    assert ranking.index(table) + 1 == 12


def test_table_chunk_search_and_rerank() -> None:
    q = "How much does the Watcher S1 weigh?"
    plain = build_pipeline(Config("bm25", "table")).retriever.search(q, 5)
    assert [h.chunk.chunk_id for h in plain].index("kb-007#4") + 1 == 4
    reranked = build_pipeline(Config("bm25", "table", rerank=True)).retriever.search(q, 5)
    assert (reranked[0].chunk.chunk_id, reranked[0].score) == ("kb-007#4", 0.75)


# --------------------------------------------------------------------------- #
# Knowledge graph, embeddings, hybrid                                         #
# --------------------------------------------------------------------------- #


def test_knowledge_graph_size() -> None:
    docs, _ = prepare_corpus(load_kb())
    kg = KnowledgeIndex(extract_triples(docs))
    assert (len(kg.triples), len(kg.by_subject)) == (90, 6)


def test_hashing_similarities() -> None:
    emb = HashingEmbedder()

    def cos(a: str, b: str) -> float:
        va, vb = emb.embed([a, b])
        return round(float(va @ vb), 2)

    assert cos("price", "cost") == 0.0
    assert cos("weigh", "weight") == 0.62
    assert cos("Mexico", "México") == 0.43
    assert cos("return window", "plazo de devolución") == -0.11


def test_rrf_worked_example() -> None:
    chunks = build_pipeline(Config("bm25", "table")).chunks
    bm, vec = BM25Retriever(chunks, fold=True), VectorRetriever(chunks)
    hybrid = HybridRetriever([bm, vec])
    q = "Who pays customs fees for orders shipped to Mexico?"
    bm_ids = [h.chunk.chunk_id for h in bm.search(q, 50)]
    vec_ids = [h.chunk.chunk_id for h in vec.search(q, len(chunks))]
    fused = hybrid.search(q, 50)
    fused_ids = [h.chunk.chunk_id for h in fused]
    assert bm_ids.index("kb-031#0") + 1 == 2
    assert vec_ids.index("kb-031#0") + 1 == 98  # beyond the 50 candidates that fusion reads
    pos = fused_ids.index("kb-031#0")
    assert (pos + 1, round(fused[pos].score, 4)) == (31, round(1 / 62, 4)) == (31, 0.0161)
    assert evaluate(build_pipeline(Config("hybrid", "table", fold=True)).retriever).by_category[
        "cross_lingual"
    ] == 0.0


def test_tokenizer_keeps_accents_by_default() -> None:
    assert tokenize("México") == ["méxico"] and tokenize("México", fold=True) == ["mexico"]


# --------------------------------------------------------------------------- #
# Exact vs approximate search (`uv run lab ann`)                              #
# --------------------------------------------------------------------------- #

# (method, params): (recall@10, compared per query, memory MB)
ANN_TABLE = {
    ("numpy exact", "-"): (1.000, 50_000, 12.80),
    ("numpy IVF", "nlist=256 nprobe=1"): (0.750, 484, 13.27),
    ("numpy IVF", "nlist=256 nprobe=8"): (0.917, 1_825, 13.27),
    ("faiss IndexFlatIP", "-"): (1.000, 50_000, 12.80),
    ("faiss IndexIVFFlat", "nlist=256 nprobe=1"): (0.741, 474, 13.27),
    ("faiss IndexIVFFlat", "nlist=256 nprobe=8"): (0.917, 1_814, 13.27),
    ("faiss IndexIVFFlat", "nlist=256 nprobe=32"): (0.977, 6_504, 13.27),
    ("faiss IndexHNSWFlat", "M=32 efSearch=16"): (0.819, 649, 26.41),
    ("faiss IndexHNSWFlat", "M=32 efSearch=64"): (0.951, 1_681, 26.41),
    ("faiss IndexHNSWFlat", "M=32 efSearch=128"): (0.981, 3_055, 26.41),
    ("faiss IndexIVFPQ", "nlist=256 m=16 nprobe=8"): (0.593, 1_814, 1.33),
}


HAS_FAISS = ann_module.faiss is not None


@pytest.fixture(scope="module")
def ann() -> dict[tuple[str, str], AnnResult]:
    return {(r.method, r.params): r for r in run_benchmark()}


def test_ann_table(ann: dict[tuple[str, str], AnnResult]) -> None:
    expected = {k: v for k, v in ANN_TABLE.items() if HAS_FAISS or not k[0].startswith("faiss")}
    assert list(ann) == list(expected)
    for key, (recall, compared, mb) in expected.items():
        r = ann[key]
        assert r.recall_at_10 == pytest.approx(recall, abs=0.02), key
        assert r.compared_per_query == pytest.approx(compared, rel=0.05), key
        assert r.memory_bytes / 1e6 == pytest.approx(mb, rel=0.01), key


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu is not installed")
def test_ann_ratios_quoted_in_lesson(ann: dict[tuple[str, str], AnnResult]) -> None:
    flat = ann[("faiss IndexFlatIP", "-")]
    hnsw = ann[("faiss IndexHNSWFlat", "M=32 efSearch=64")]
    pq = ann[("faiss IndexIVFPQ", "nlist=256 m=16 nprobe=8")]
    ivf8 = ann[("faiss IndexIVFFlat", "nlist=256 nprobe=8")]
    assert round(hnsw.memory_bytes / flat.memory_bytes, 1) == pytest.approx(2.1, abs=0.1)
    assert round(flat.memory_bytes / pq.memory_bytes) == pytest.approx(10, abs=1)
    assert round(flat.compared_per_query / ivf8.compared_per_query) == pytest.approx(28, abs=2)
    assert 50_000 * 64 * 4 == 12_800_000  # flat index: n x dim x 4 bytes
    ivf32 = ann[("faiss IndexIVFFlat", "nlist=256 nprobe=32")]
    assert round(ivf32.compared_per_query / ivf8.compared_per_query, 1) == pytest.approx(3.6, abs=0.2)
    assert 64 * 4 == 256 and 16 * 1 == 16  # bytes per vector: raw float32 vs 16 PQ codes


def test_benchmark_setup() -> None:
    base, queries = make_vectors()
    assert base.shape == (50_000, 64) and queries.shape == (200, 64)


def test_break_it_destroyed_clusters() -> None:
    rows = {r.params: r.recall_at_10 for r in run_benchmark(use_faiss=False, spread=3.0)}
    assert rows["nlist=256 nprobe=1"] == pytest.approx(0.069, abs=0.02)
    assert rows["nlist=256 nprobe=8"] == pytest.approx(0.301, abs=0.02)


# --------------------------------------------------------------------------- #
# Numbers quoted in lesson 5.1, 5.3, the lab README and the quiz               #
# --------------------------------------------------------------------------- #


def test_unfiltered_exposure_breakdown() -> None:
    pipe = build_pipeline(Config("bm25", "heading", filter_meta=False))
    old = internal = 0
    from m05_retrieval.evaluate import doc_ranking

    for q in load_questions(answerable_only=False):
        top = doc_ranking(pipe.retriever.search(q["question"], 50))[:5]
        old += "kb-010" in top
        internal += "kb-041" in top
    assert (old, internal) == (11, 4)
    hit = pipe.retriever.search("Can support give me a goodwill credit code?", 1)[0]
    assert (hit.chunk.chunk_id, round(hit.score, 3)) == ("kb-041#1", 20.474)


def test_q004_slips_from_fifth_to_sixth() -> None:
    from m05_retrieval.evaluate import doc_ranking

    q = "How long is the standard warranty on Hearthwise devices?"
    raw = doc_ranking(build_pipeline(Config("bm25", "heading", filter_meta=False)).retriever.search(q, 50))
    clean = doc_ranking(build_pipeline(Config("bm25", "heading")).retriever.search(q, 50))
    assert (raw.index("kb-012") + 1, clean.index("kb-012") + 1) == (5, 6)


def test_prompt_size() -> None:
    from m05_retrieval.prompting import build_messages

    pipe = build_pipeline(Config("bm25", "table"))
    q = "Can I return an opened Glow A19?"
    msgs = build_messages(q, pipe.retriever.search(q, 3))
    assert sum(count_tokens(m.content) for m in msgs) == 207


def test_manual_has_13_headings() -> None:
    manual = next(d for d in load_kb() if d.doc_id == "kb-040")
    assert sum(b.kind == "heading" for b in manual.blocks) == 13


def test_saturation_and_rrf_arithmetic() -> None:
    k1 = 1.5
    assert [round(tf * (k1 + 1) / (tf + k1), 2) for tf in (1, 2, 10)] == [1.0, 1.43, 2.17]
    assert k1 + 1 == 2.5
    assert (round(1 / 61, 4), round(2 / 70, 4)) == (0.0164, 0.0286)


def test_boolean_reset_and_hub_is_empty() -> None:
    idx = BM25Retriever(build_pipeline(Config("bm25", "table")).chunks).bm25.index
    assert (len(idx.postings["reset"]), len(idx.postings["hub"])) == (16, 21)
    assert idx.boolean_and(["reset", "hub"]) == set()


def test_vector_snippet_in_lesson_3() -> None:
    texts = ["Price: €29", "How much does it cost?", "Weight (g): 34"]
    hits = VectorIndex(texts, HashingEmbedder()).search("how much does it weigh", k=2)
    assert [(i, round(s, 2)) for i, s in hits] == [(1, 0.78), (2, 0.23)]


def test_index_ratio_and_vector_bytes() -> None:
    assert 256 * 4 == 1024 and 128 * 1024 == 131072
    assert round(131072 / 32064) == 4


def test_reranker_candidate_budget() -> None:
    assert RerankedRetriever(BM25Retriever(build_pipeline(Config("bm25", "table")).chunks)).candidates == 20


# --------------------------------------------------------------------------- #
# Break it / Extend it (lab README)                                           #
# --------------------------------------------------------------------------- #


def test_break_it_filters_off_with_table_chunks() -> None:
    r = evaluate(build_pipeline(Config("bm25", "table", filter_meta=False)).retriever)
    assert (r.recall_at_5, r.mrr, r.blocked_in_top5) == (0.944, 0.758, 15)


def test_break_it_no_length_normalisation() -> None:
    pipe = build_pipeline(Config("bm25", "heading"))
    pipe.retriever.bm25.b = 0.0
    r = evaluate(pipe.retriever)
    assert (r.recall_at_5, r.mrr, r.by_category["table"], r.misses) == (0.944, 0.843, 1.0, ["q021", "q022"])


def test_break_it_query_tokenizer_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import m05_retrieval.bm25 as bm25_module

    def mismatched(text: str, *, fold: bool = False) -> list[str]:
        return text.split()  # no lower-casing, punctuation kept: "weigh?" never matches "weigh"

    pipe = build_pipeline(Config("bm25", "table"))  # the index is built with the real tokenizer
    monkeypatch.setattr(bm25_module, "tokenize", mismatched)  # ... and only queries use this one
    r = evaluate(pipe.retriever)
    assert (r.recall_at_5, r.mrr, len(r.misses)) == (0.861, 0.739, 5)


def test_extend_it_title_prefix() -> None:
    r = evaluate(build_pipeline(Config("bm25", "table", title=True)).retriever)
    assert (r.recall_at_1, r.recall_at_5, r.mrr, r.misses) == (0.722, 0.972, 0.824, ["q022"])


def test_lesson_examples_for_how_questions() -> None:
    pipe = build_pipeline(Config("bm25", "heading"))
    bm = pipe.retriever.bm25
    df_how = len(bm.index.postings["how"])
    assert (df_how, round(idf(bm.index.n_docs, df_how), 2)) == (7, 2.70)
    top = pipe.retriever.search("How do I factory reset the Ember T3?", 2)
    assert [h.chunk.chunk_id for h in top] == ["kb-026#1", "kb-026#2"]
    e2 = pipe.retriever.search("What does error code E2 mean on the Ember T3?", 1)[0]
    assert (e2.chunk.chunk_id, round(e2.score, 2)) == ("kb-040#10", 11.04)


def test_quiz_idf_arithmetic() -> None:
    assert (111 - 4 + 0.5, 4 + 0.5, round(1 + 107.5 / 4.5, 2)) == (107.5, 4.5, 24.89)


# --------------------------------------------------------------------------- #
# Expected-output blocks in the Markdown match a fresh run                   #
# --------------------------------------------------------------------------- #

MODULE_DIR = Path(__file__).resolve().parents[2]
IGNORED = ("varies by machine",)


def _fenced_blocks(path: Path) -> list[tuple[str, list[str], int, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks, i = [], 0
    while i < len(lines):
        m = re.match(r"^(\s*)```(\w*)\s*$", lines[i])
        if not m:
            i += 1
            continue
        indent, lang, j = m.group(1), m.group(2), i + 1
        while not re.match(rf"^{indent}```\s*$", lines[j]):
            j += 1
        blocks.append((lang, [ln[len(indent):] for ln in lines[i + 1 : j]], i, j))
        i = j + 1
    return blocks


def _command_output_pairs() -> list[tuple[str, str, list[str]]]:
    pages = sorted(MODULE_DIR.glob("*.md")) + [MODULE_DIR / "lab" / "README.md"]
    pairs = []
    for page in pages:
        blocks = _fenced_blocks(page)
        for (lang1, body1, _, end1), (lang2, body2, start2, _) in zip(blocks, blocks[1:]):
            single = len(body1) == 1 and body1[0].startswith("uv run lab ")
            is_ann = body1[0].startswith("uv run lab ann")  # asserted with tolerances in test_ann_table
            if lang1 == "bash" and lang2 == "text" and single and start2 - end1 <= 4 and not is_ann:
                pairs.append((page.name, body1[0], body2))
    return pairs


PAIRS = _command_output_pairs()


def test_markdown_has_command_output_pairs() -> None:
    assert len(PAIRS) >= 16


@pytest.mark.parametrize(("page", "command", "expected"), PAIRS, ids=[f"{p}: {c}" for p, c, _ in PAIRS])
def test_expected_output_blocks(page: str, command: str, expected: list[str],
                                capsys: pytest.CaptureFixture[str]) -> None:
    main(shlex.split(command)[3:])
    actual = [ln.rstrip() for ln in capsys.readouterr().out.splitlines()]
    pos = 0
    for line in expected:
        line = line.rstrip()
        if not line.strip() or line.strip() == "..." or any(word in line for word in IGNORED):
            continue
        assert line in actual[pos:], f"{page}: {command!r}: line not found in order: {line!r}"
        pos = actual.index(line, pos) + 1


def test_quiz_has_13_questions() -> None:
    quiz = (MODULE_DIR / "quiz.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^### Q\d+", quiz, re.M)) == 13


def test_import_duties_finds_nothing() -> None:
    pipe = build_pipeline(Config("bm25", "table", fold=True))
    assert pipe.retriever.search("Who pays import duties?", 50) == []
