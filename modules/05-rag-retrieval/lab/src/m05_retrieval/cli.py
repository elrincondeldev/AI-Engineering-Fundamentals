"""Command-line entry point: ``uv run lab <command>``."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from .ann import run_benchmark
from .chunking import CHUNKERS, near_duplicates, prepare_corpus
from .evaluate import EvalReport, evaluate, load_questions
from .knowledge import KnowledgeIndex, extract_triples
from .parsing import load_kb
from .prompting import build_messages
from .providers import count_tokens
from .retrieval import BM25Retriever, Config, Pipeline, VectorRetriever, build_pipeline
from .text import word_count
from .vectors import Embedder, HashingEmbedder, ProviderEmbedder

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"

# The configurations compared by `uv run lab eval` (in this order).
LINEUP: list[Config] = [
    Config("bm25", "heading", filter_meta=False),
    Config("bm25", "heading"),
    Config("bm25", "fixed"),
    Config("bm25", "recursive"),
    Config("bm25", "semantic"),
    Config("bm25", "table"),
    Config("vector", "table"),
    Config("hybrid", "table"),
    Config("bm25", "table", fold=True),
    Config("bm25", "table", rerank=True),
    Config("bm25", "table", fold=True, rerank=True),
]


def _config_from(args: argparse.Namespace) -> Config:
    return Config(
        retriever=args.retriever or "bm25",
        chunker=args.chunker or "table",
        filter_meta=not args.no_filter,
        title=args.title,
        fold=args.fold,
        rerank=args.rerank,
    )


def _embedder(args: argparse.Namespace) -> Embedder:
    """``--embedder provider`` uses LLM_PROVIDER's embeddings (for example Ollama); the default is offline."""
    return ProviderEmbedder() if getattr(args, "embedder", "hashing") == "provider" else HashingEmbedder()


def _pipeline(args: argparse.Namespace) -> Pipeline:
    return build_pipeline(_config_from(args), embedder=_embedder(args))


def cmd_parse(_: argparse.Namespace) -> None:
    docs = load_kb()
    print(f"{'doc':7} {'format':8} {'lang':4} {'status':10} {'audience':8} {'headings':>8} {'texts':>5} "
          f"{'tables':>6} {'hidden':>6}")
    for d in docs:
        kinds = Counter(b.kind for b in d.blocks)
        print(f"{d.doc_id:7} {d.format:8} {d.meta['lang']:4} {d.meta['status']:10} {d.meta['audience']:8} "
              f"{kinds['heading']:>8} {kinds['text']:>5} {kinds['table']:>6} {d.hidden_dropped:>6}")
    tables = sum(b.kind == "table" for d in docs for b in d.blocks)
    rows = sum(len(b.rows) - 1 for d in docs for b in d.blocks if b.kind == "table")
    hidden = sum(d.hidden_dropped for d in docs)
    tokens = sum(count_tokens(d.text) for d in docs)
    print(f"\n{len(docs)} documents, {tables} tables ({rows} data rows), "
          f"{hidden} hidden HTML element(s) dropped")
    print(f"about {tokens:,} tokens of parsed text (rough count: words and punctuation)")


def cmd_chunk(args: argparse.Namespace) -> None:
    config = _config_from(args)
    docs = load_kb()
    for kept, dup, sim in near_duplicates(docs):
        print(f"near-duplicate: {dup} ~ {kept} (Jaccard {sim})")
    pipe = build_pipeline(config, docs)
    for doc_id, reason in sorted(pipe.report.dropped.items()):
        print(f"dropped {doc_id}: {reason}")
    sizes = [word_count(c.text) for c in pipe.chunks]
    print(f"\nchunker={config.chunker}: {len(pipe.report.kept)} documents -> {len(pipe.chunks)} chunks "
          f"(words per chunk: mean {statistics.mean(sizes):.1f}, max {max(sizes)})")
    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / f"chunks-{config.chunker}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for c in pipe.chunks:
            fh.write(json.dumps({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "section": c.section,
                                 "text": c.text}, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(OUTPUTS.parent)}")
    if args.show:
        for c in pipe.chunks:
            if c.doc_id == args.show:
                print(f"\n--- {c.chunk_id} [{c.section}]\n{c.text}")


def cmd_index(args: argparse.Namespace) -> None:
    config = _config_from(args)
    pipe = build_pipeline(Config("bm25", config.chunker, filter_meta=config.filter_meta))
    texts = [c.text for c in pipe.chunks]
    t0 = time.perf_counter()
    bm = BM25Retriever(pipe.chunks)
    t_bm = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    vec = VectorRetriever(pipe.chunks)
    t_vec = (time.perf_counter() - t0) * 1000
    idx = bm.bm25.index
    raw_bytes = sum(len(t.encode("utf-8")) for t in texts)
    inv_bytes = idx.n_postings * 8 + sum(len(t.encode("utf-8")) for t in idx.postings)
    print(f"chunks:              {len(texts)}")
    print(f"chunk text:          {raw_bytes:,} bytes")
    print(f"vocabulary:          {len(idx.postings):,} terms")
    print(f"postings:            {idx.n_postings:,} (term, chunk) pairs")
    print(f"inverted index:      ~{inv_bytes:,} bytes (8 bytes per posting + the terms)")
    print(f"vector matrix:       {vec.index.nbytes:,} bytes ({len(texts)} x 256 float32)")
    questions = [q["question"] for q in load_questions()]
    lat = {}
    for name, r in (("bm25", bm), ("vector", vec)):
        t0 = time.perf_counter()
        for q in questions:
            r.search(q, 5)
        lat[name] = (time.perf_counter() - t0) * 1000 / len(questions)
    print(f"\nbuild time (example, varies by machine): bm25 {t_bm:.1f} ms, vector {t_vec:.1f} ms")
    print(f"query time (example, varies by machine): bm25 {lat['bm25']:.2f} ms, "
          f"vector {lat['vector']:.2f} ms")


def cmd_search(args: argparse.Namespace) -> None:
    config = _config_from(args)
    pipe = _pipeline(args)
    t0 = time.perf_counter()
    hits = pipe.retriever.search(args.query, args.k)
    ms = (time.perf_counter() - t0) * 1000
    print(f"config: {config.name}  ({len(pipe.chunks)} chunks)")
    for rank, h in enumerate(hits, start=1):
        preview = " ".join(h.chunk.text.split())[:90]
        print(f"{rank}. {h.chunk.chunk_id:9} score={h.score:.3f}  {preview}")
    print(f"query time: {ms:.2f} ms (example, varies by machine)")


def _row(r: EvalReport, chunks: int) -> str:
    return (f"{r.name:28} {chunks:>6} {r.recall_at_1:>5.3f} {r.recall_at_5:>5.3f} {r.mrr:>5.3f} "
            f"{r.context_at_5:>5.3f} "
            f"{r.by_category.get('table', 0):>5.2f} {r.by_category.get('cross_lingual', 0):>6.2f} "
            f"{r.blocked_in_top5:>7}")


def run_lineup() -> list[tuple[Config, int, EvalReport]]:
    docs = load_kb()
    out = []
    for config in LINEUP:
        pipe = build_pipeline(config, docs)
        out.append((config, len(pipe.chunks), evaluate(pipe.retriever, config.name)))
    return out


def cmd_eval(args: argparse.Namespace) -> None:
    single = any([args.retriever, args.chunker, args.no_filter, args.title, args.fold, args.rerank])
    header = (f"{'config':28} {'chunks':>6} {'R@1':>5} {'R@5':>5} {'MRR':>5} {'ctx@5':>5} {'table':>5} "
              f"{'x-ling':>6} {'exposed':>7}")
    if not single:
        print(f"{len(load_questions())} answerable questions. R@k, MRR: document ranking. "
              "ctx@5: first 5 chunks.\ntable, x-ling: R@5 per category. exposed: questions (of 40) "
              "with a superseded or internal doc in the top 5.\n")
        print(header)
        for config, chunks, report in run_lineup():
            print(_row(report, chunks))
        return
    config = _config_from(args)
    pipe = _pipeline(args)
    report = evaluate(pipe.retriever, config.name)
    print(header)
    print(_row(report, len(pipe.chunks)))
    print("\nrecall@5 by category:")
    for cat, value in report.by_category.items():
        print(f"  {cat:18} {value:.2f}")
    print(f"misses at 5: {', '.join(report.misses) or 'none'}")


def cmd_prompt(args: argparse.Namespace) -> None:
    pipe = _pipeline(args)
    messages = build_messages(args.query, pipe.retriever.search(args.query, args.k))
    for m in messages:
        print(f"--- {m.role}\n{m.content}")
    print(f"\n~{sum(count_tokens(m.content) for m in messages)} tokens (rough count)")


def cmd_kg(args: argparse.Namespace) -> None:
    docs, _ = prepare_corpus(load_kb())
    triples = extract_triples(docs)
    kg = KnowledgeIndex(triples)
    print(f"{len(triples)} triples about {len(kg.by_subject)} products")
    found = kg.lookup(args.query)
    if not found:
        print("no matching fact (fall back to text search)")
    for t in found:
        print(f"({t.subject}, {t.predicate}, {t.object})  source: {t.doc_id}")


def cmd_ann(args: argparse.Namespace) -> None:
    results = run_benchmark(n=args.n, use_faiss=not args.no_faiss, spread=args.spread)
    print(f"{args.n:,} vectors x 64 dims, 200 queries, recall@10 against exact search\n")
    print(f"{'method':20} {'params':25} {'recall@10':>9} {'compared':>9} {'memory MB':>9} {'ms/query*':>9}")
    for r in results:
        print(f"{r.method:20} {r.params:25} {r.recall_at_10:>9.3f} {r.compared_per_query:>9,} "
              f"{r.memory_bytes / 1e6:>9.2f} {r.ms_per_query:>9.3f}")
    print("\n* example timings, vary by machine. compared = similarities computed per query")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab", description="Lab 05: retrieval for the Hearthwise help centre."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--retriever", choices=["bm25", "vector", "hybrid"])
        p.add_argument("--chunker", choices=sorted(CHUNKERS))
        p.add_argument("--no-filter", action="store_true", help="keep superseded and internal documents")
        p.add_argument("--title", action="store_true", help="prefix chunks with the document title")
        p.add_argument("--fold", action="store_true", help="fold accents in BM25 (México == Mexico)")
        p.add_argument("--rerank", action="store_true", help="rerank the top 20 with the lexical reranker")
        p.add_argument("--embedder", choices=["hashing", "provider"], default="hashing",
                       help="vector embeddings: offline hashing (default) or LLM_PROVIDER's embed()")

    sub.add_parser("parse", help="parse every help-centre document and show its structure")
    p = sub.add_parser("chunk", help="filter, de-duplicate and chunk the corpus")
    add_config(p)
    p.add_argument("--show", metavar="DOC_ID", help="print the chunks of one document")
    p = sub.add_parser("index", help="index sizes and example latencies")
    add_config(p)
    p = sub.add_parser("search", help="search the help centre")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    add_config(p)
    p = sub.add_parser("eval", help="doc recall@k and MRR on rag_eval (no flags: compare the lineup)")
    add_config(p)
    p = sub.add_parser("prompt", help="show the grounded prompt a generator would receive")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=3)
    add_config(p)
    p = sub.add_parser("kg", help="look a product fact up in the knowledge graph")
    p.add_argument("query")
    p = sub.add_parser("ann", help="exact vs approximate nearest-neighbour search (FAISS)")
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--no-faiss", action="store_true", help="numpy only")
    p.add_argument("--spread", type=float, default=1.5,
                   help="noise around each topic centre (larger = less clustered, harder for ANN)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    handlers = {"parse": cmd_parse, "chunk": cmd_chunk, "index": cmd_index, "search": cmd_search,
                "eval": cmd_eval, "prompt": cmd_prompt, "kg": cmd_kg, "ann": cmd_ann}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
