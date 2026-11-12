"""Command-line entry point: ``uv run lab <command>``."""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import bpe, cleaning, ngram, params, published, transformer
from .data import iter_csv
from .dedup import MinHasher, estimated_jaccard, jaccard, lsh_threshold, shingles
from .neural import train_neural_bigram
from .providers import Message, get_provider

OUTPUTS = Path("outputs")


@lru_cache(maxsize=1)
def _default_clean() -> tuple[list[cleaning.Doc], cleaning.Report]:
    return cleaning.clean_crawl()


@lru_cache(maxsize=4)
def default_tokenizer(vocab_size: int = 1024, es_weight: int = 1) -> bpe.BPETokenizer:
    """The tokenizer every command shares: byte-level BPE trained on the cleaned crawl."""
    docs, _ = _default_clean()
    texts = cleaning.kept_texts(docs) + cleaning.kept_texts(docs, "es") * (es_weight - 1)
    tok, _ = bpe.train_bpe(texts, vocab_size)
    return tok


def ticket_bodies() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"en": [], "es": []}
    for row in iter_csv("tickets/tickets.csv"):
        out[row["language"]].append(row["body"])
    return out


# --------------------------------------------------------------------------- #
# Commands                                                                    #
# --------------------------------------------------------------------------- #


def cmd_ngram(args: argparse.Namespace) -> None:
    docs, _ = _default_clean()
    texts = cleaning.kept_texts(docs, "en")
    model = ngram.NGramModel(order=4).fit(ngram.corpus_tokens(texts))
    print(f"Trained a 4-gram counting model on {len(texts)} cleaned English pages.\n")
    for prompt in args.prompt:
        used, probs = model.next_word_probs(ngram.words(prompt))
        print(f"Prompt: {prompt!r}  (matched the last {used} word(s))")
        for word, p in probs:
            print(f"  {word:<12} {p:.3f}")
        print(f"  greedy continuation: {' '.join(model.generate(prompt))}\n")
    train, held = ngram.split_docs(texts)
    tr, ho = ngram.corpus_tokens(train), ngram.corpus_tokens(held)
    print(f"Held-out n-grams already seen in training ({len(train)} train / {len(held)} held-out pages):")
    for n in range(1, 6):
        print(f"  n={n}: {ngram.coverage(tr, ho, n):.3f}")


def cmd_clean(args: argparse.Namespace) -> None:
    skip = set(args.skip or [])
    cfg = cleaning.PipelineConfig(
        quality=cleaning.QualityConfig(min_words=args.min_words,
                                       max_top_trigram_share=args.max_trigram_share),
        use_url_dedup="url_dedup" not in skip,
        use_exact_dedup="exact_dup" not in skip,
        use_near_dedup="near_dup" not in skip,
        use_decontamination="contamination" not in skip,
    )
    docs, rep = cleaning.clean_crawl(cfg)
    ok = sum(d.status == 200 for d in docs)
    html = sum(d.content_type == "text/html" for d in docs)
    print(f"Crawl: {rep.total} pages ({ok} with HTTP 200, {rep.total - ok} errors; "
          f"{html} text/html, {rep.total - html} other types). Labels say keep {rep.tp + rep.fn}.\n")
    print("Dropped by stage:")
    remaining = rep.total
    for stage, n in rep.dropped_by_stage.items():
        remaining -= n
        print(f"  {stage:<14} -{n:<3} -> {remaining}")
    print(f"\nKept {rep.kept} pages: {rep.text_chars_kept:,} characters of text "
          f"from {rep.html_chars:,} characters of raw HTML ({rep.text_chars_kept / rep.html_chars:.1%}).")
    langs = {lang: sum(1 for d in docs if d.kept and d.lang == lang) for lang in cfg.keep_langs}
    print("Languages kept: " + ", ".join(f"{k}={v}" for k, v in langs.items()))
    pii = ", ".join(f"{k}={v}" for k, v in rep.pii_counts.items()) or "none"
    print(f"PII redacted on {rep.pii_pages} pages: {pii}")
    print("\nAgainst labels.jsonl (positive class = keep):")
    print(f"  TP={rep.tp}  FP={rep.fp}  FN={rep.fn}  TN={rep.tn}")
    print(f"  precision={rep.precision:.3f}  recall={rep.recall:.3f}  F1={rep.f1:.3f}")
    for url in rep.false_keeps[: args.show]:
        print(f"  kept but should drop: {url}")
    for url, reason in rep.false_drops[: args.show]:
        print(f"  dropped but should keep: {url}  ({reason})")
    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / "clean_corpus.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for d in sorted(docs, key=lambda d: d.url):
            if d.kept:
                row = {"url": d.url, "lang": d.lang, "text": d.text}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {out}")


def cmd_minhash(args: argparse.Namespace) -> None:
    docs, _ = cleaning.clean_crawl(cleaning.PipelineConfig(use_near_dedup=False))
    by_url = {d.url: d for d in docs}
    hasher = MinHasher(num_perm=args.num_perm)
    print(f"MinHash with {args.num_perm} hash functions; LSH with 16 bands of {args.num_perm // 16} rows "
          f"(50% candidate threshold ~ {lsh_threshold(16, args.num_perm // 16):.3f})\n")
    print(f"{'pair':<58} {'exact J':>8} {'MinHash':>8}")
    from .data import read_jsonl  # local import keeps the command self-explanatory

    for row in read_jsonl("web/labels.jsonl"):
        if not row["near_duplicate_of"]:
            continue
        a, b = by_url[row["url"]], by_url[row["near_duplicate_of"]]
        sa, sb = shingles(a.text), shingles(b.text)
        est = estimated_jaccard(hasher.signature(sa), hasher.signature(sb))
        name = row["url"].split("/")[-1][:56]
        print(f"{name:<58} {jaccard(sa, sb):>8.3f} {est:>8.3f}")
    kept = [d for d in docs if d.kept and not d.url.startswith("https://newsmirror.example/copy/")]
    pairs = [(jaccard(shingles(x.text), shingles(y.text)), x.url, y.url)
             for i, x in enumerate(kept) for y in kept[i + 1:]]
    best = max(pairs)
    print(f"\nMost similar pair of different articles: J={best[0]:.3f}\n  {best[1]}\n  {best[2]}")


def cmd_bpe(args: argparse.Namespace) -> None:
    docs, _ = _default_clean()
    texts = cleaning.kept_texts(docs) + cleaning.kept_texts(docs, "es") * (args.es_weight - 1)
    tok, history = bpe.train_bpe(texts, args.vocab_size)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    print(f"Trained byte-level BPE on {len(texts)} documents ({n_bytes:,} bytes): "
          f"{len(tok.merges)} merges + 256 bytes + {len(tok.special)} special = vocab {tok.vocab_size}\n")
    print("First 10 merges:")
    for m in history[:10]:
        left, right = (tok.vocab[i].decode("utf-8", "replace").replace(" ", "·") for i in m.pair)
        print(f"  #{m.rank:<3} {left!r:>8} + {right!r:<8} -> {m.piece.decode('utf-8', 'replace')!r:<10} "
              f"(seen {m.count} times)")
    for text in args.text:
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        print(f"\n{text!r}\n  {len(ids)} tokens: {' | '.join(tok.token_strings(ids))}")
    special = tok.encode(f"Hola{bpe.END_OF_TEXT}Hello")
    print(f"\nSpecial token: 'Hola{bpe.END_OF_TEXT}Hello' -> {special}")
    tickets = ticket_bodies()
    print(f"\nTokens per word on Hearthwise tickets ({len(tickets['en'])} en / {len(tickets['es'])} es):")
    print(f"  {'vocab':>6} {'en':>7} {'es':>7} {'es/en':>7}")
    for size in (257, 384, 512, 768, 1024):
        if size > tok.vocab_size:
            break
        t = tok.truncated(size - 257)
        en, es = bpe.tokens_per_word(t, tickets["en"]), bpe.tokens_per_word(t, tickets["es"])
        print(f"  {t.vocab_size:>6} {en:>7.3f} {es:>7.3f} {es / en:>7.3f}")


def cmd_attention(args: argparse.Namespace) -> None:
    tok = default_tokenizer()
    cfg = transformer.TinyConfig(vocab_size=tok.vocab_size)
    p = transformer.init_params(cfg, seed=args.seed)
    ids = np.array(tok.encode(args.text))
    pieces = tok.token_strings(ids.tolist())
    t = len(ids)
    print(f"Text: {args.text!r}\nTokens ({t}): {' | '.join(pieces)}\n")
    print(f"Config: vocab={cfg.vocab_size} d_model={cfg.d_model} heads={cfg.n_heads} "
          f"d_head={cfg.d_head} layers={cfg.n_layers} d_ff={cfg.d_ff}")
    x = p["tok_emb"][ids]
    h = transformer.layer_norm(x, p["l0.ln1.g"], p["l0.ln1.b"])
    q = transformer.split_heads(h @ p["l0.wq"], cfg.n_heads)
    scores = q @ np.swapaxes(transformer.split_heads(h @ p["l0.wk"], cfg.n_heads), -1, -2)
    print(f"  token ids          {ids.shape}")
    print(f"  embeddings x       {x.shape}")
    print(f"  queries per head   {q.shape}   (heads, tokens, d_head)")
    print(f"  scores Q.K^T       {scores.shape}")
    logits, weights = transformer.forward(ids, p, cfg)
    print(f"  attention weights  {weights[0].shape}")
    print(f"  logits             {logits.shape}   (one score per vocabulary entry, per position)\n")
    print("Layer 0, head 0 attention weights (row = query token, column = key token):")
    w = weights[0][0]
    width = max(len(s) for s in pieces) + 1
    print(" " * width + "".join(f"{s[:6]:>7}" for s in pieces))
    for i, s in enumerate(pieces):
        print(f"{s:<{width}}" + "".join(f"{w[i, j]:>7.2f}" for j in range(t)))
    print(f"\nEvery row sums to 1: {bool(np.allclose(w.sum(axis=1), 1.0))}; "
          f"weights above the diagonal: {float(np.triu(w, 1).sum()):.1f}")
    loss = transformer.cross_entropy(logits[:-1], ids[1:])
    print(f"Next-token loss with random weights: {loss:.3f} (uniform guess: ln {cfg.vocab_size} = "
          f"{math.log(cfg.vocab_size):.3f})")


def cmd_train(args: argparse.Namespace) -> None:
    tok = default_tokenizer()
    docs, _ = _default_clean()
    train_docs, valid_docs = ngram.split_docs(cleaning.kept_texts(docs))
    enc = lambda ts: np.array([i for t in ts for i in tok.encode(t + bpe.END_OF_TEXT)])  # noqa: E731
    train, valid = enc(train_docs), enc(valid_docs)
    print(f"Neural bigram: embedding {tok.vocab_size}x{args.dim} -> logits {args.dim}x{tok.vocab_size}; "
          f"{len(train):,} train / {len(valid):,} validation tokens; lr={args.lr}\n")
    res = train_neural_bigram(train, valid, tok.vocab_size, dim=args.dim, steps=args.steps, lr=args.lr,
                              eval_every=max(1, args.steps // 5))
    print(f"{'step':>6} {'train':>8} {'valid':>8}")
    for step, tr, va in res.history:
        print(f"{step:>6} {tr:>8.3f} {va:>8.3f}")
    print(f"\nUniform guess:            {res.uniform_loss:.3f}")
    print(f"Counting bigram (smoothed): {res.count_bigram_loss:.3f}")


def cmd_params(args: argparse.Namespace) -> None:
    print(f"{'preset':<20} {'total':>15} {'active':>15} {'bf16 GB':>8} {'KV MB/1k tok':>13}")
    for s in params.PRESETS.values():
        c = params.count_params(s)
        kv = params.kv_cache_bytes(s, 1000) / 1e6
        gb = params.weight_memory_gb(c.total)
        print(f"{s.name:<20} {c.total:>15,} {c.active:>15,} {gb:>8.2f} {kv:>13.2f}")
    s = params.PRESETS[args.show]
    c = params.count_params(s)
    print(f"\nBreakdown of {s.name}:")
    for k, v in c.breakdown.items():
        print(f"  {k:<14} {v:>15,}  {v / c.total:>6.1%}")
    if not s.is_moe and s.ffn == "mlp" and s.d_ff == 4 * s.d_model:
        rule = params.kaplan_estimate(s)
        print(f"  non-embedding  {c.non_embedding:>15,}  (Kaplan rule 12*L*d^2 = {rule:,})")
    else:
        print(f"  non-embedding  {c.non_embedding:>15,}  "
              "(the 12*L*d^2 rule only fits dense MLPs with d_ff = 4*d)")
    fp32, bf16, int4 = (params.weight_memory_gb(c.total, d) for d in ("fp32", "bf16", "int4"))
    print(f"  weights: fp32 {fp32:.2f} GB, bf16 {bf16:.2f} GB, int4 {int4:.2f} GB")
    tokens = args.tokens
    flops = params.training_flops(c.active, tokens)
    print(f"  training compute for {tokens:.0e} tokens ~ 6*N*D = {flops:.2e} FLOPs")
    full = params.kv_cache_bytes(s, s.context_length)
    local = params.kv_cache_bytes(s, s.context_length, local_window=1024, local_per_global=5)
    print(f"  KV cache at {s.context_length:,} tokens: {full / 1e9:.3f} GB all-global, "
          f"{local / 1e9:.3f} GB with 5 local (1024-token window) per global layer")
    print("\nPublished mixture-of-experts sizes (sources: uv run lab facts):")
    for name, total, active in (("DeepSeek-V3", "total", "parameters activated"),
                                ("Mixtral 8x7B", "parameters each", "active")):
        t, a = published.fact(name, total), published.fact(name, active)
        print(f"  {name:<14} total {t:g}B, active {a:g}B per token -> {params.active_share(t, a):.1%} active")


def cmd_facts(args: argparse.Namespace) -> None:
    print("Figures quoted from papers (not lab results). Check them at the source:\n")
    for f in published.FACTS:
        if args.topic and args.topic.lower() not in f.topic.lower():
            continue
        print(f"  {f.topic:<13} {f.claim:<52} {f.value:>9,g} {f.unit:<16} {f.source}")


def _base_responder(messages) -> str:  # noqa: ANN001 - signature fixed by FakeProvider
    """A 'base model' stand-in: continue the text with the counting model instead of answering."""
    docs, _ = _default_clean()
    model = ngram.NGramModel(order=4).fit(ngram.corpus_tokens(cleaning.kept_texts(docs, "en")))
    return " ".join(model.generate(messages[-1].content, max_words=15))


def cmd_ask(args: argparse.Namespace) -> None:
    question = args.question
    base = get_provider("fake", responder=_base_responder)
    print(f"Question: {question}\n")
    print(f"Base-model style (4-gram continuation): ...{base.complete([Message('user', question)]).text}")
    chat = get_provider()
    reply = chat.complete([Message("system", "You are Hearthwise's helpful support assistant."),
                           Message("user", question)], max_tokens=200)
    print(f"Assistant via LLM_PROVIDER={chat.name} ({reply.model}): {reply.text}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lab", description="Module 01 lab: how LLMs are built.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ngram", help="lesson 1.1: a counting next-word predictor")
    p.add_argument("--prompt", action="append",
                   help="prompt to continue (repeatable)")
    p.set_defaults(func=cmd_ngram)

    p = sub.add_parser("ask", help="lesson 1.1: base-model continuation vs assistant answer")
    p.add_argument("question", nargs="?", default="How do I reset my thermostat?")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("clean", help="lesson 1.2: run the pre-training cleaning pipeline")
    p.add_argument("--min-words", type=int, default=cleaning.QualityConfig.min_words)
    p.add_argument("--max-trigram-share", type=float, default=cleaning.QualityConfig.max_top_trigram_share)
    p.add_argument("--skip", action="append", choices=["url_dedup", "exact_dup", "near_dup", "contamination"])
    p.add_argument("--show", type=int, default=5, help="how many errors to list")
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("minhash", help="lesson 1.2: exact vs MinHash-estimated Jaccard similarity")
    p.add_argument("--num-perm", type=int, default=128)
    p.set_defaults(func=cmd_minhash)

    p = sub.add_parser("bpe", help="lesson 1.3: train a byte-level BPE tokenizer")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--es-weight", type=int, default=1, help="repeat Spanish pages this many times")
    p.add_argument("--text", action="append", help="text to tokenize (repeatable)")
    p.set_defaults(func=cmd_bpe)

    p = sub.add_parser("attention", help="lesson 1.4: causal self-attention in numpy, with shapes")
    p.add_argument("--text", default="Lower the setpoint by one degree")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_attention)

    p = sub.add_parser("train", help="lesson 1.4: gradient descent on a neural bigram model")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=10.0)
    p.add_argument("--dim", type=int, default=32)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("params", help="lesson 1.5: parameters, memory and compute, dense vs MoE")
    p.add_argument("--show", choices=sorted(params.PRESETS), default="moe-8x-top2")
    p.add_argument("--tokens", type=float, default=1e12, help="training tokens for the compute estimate")
    p.set_defaults(func=cmd_params)

    p = sub.add_parser("facts", help="published figures quoted in the lessons, with sources")
    p.add_argument("--topic", default="", help="filter, for example DeepSeek")
    p.set_defaults(func=cmd_facts)

    args = ap.parse_args(argv)
    if args.command == "ngram" and not args.prompt:
        args.prompt = ["Lower the setpoint by one degree in the", "The Ember T3"]
    if args.command == "bpe" and not args.text:
        args.text = ["How long does shipping take?", "¿Cuánto tarda el envío?"]
    args.func(args)


if __name__ == "__main__":
    main()
