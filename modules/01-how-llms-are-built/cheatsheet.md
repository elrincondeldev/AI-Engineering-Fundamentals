🌐 English · [Español](cheatsheet.es.md)

# Module 01 cheatsheet — How LLMs are built

## The path

```text
raw crawl → cleaning → tokenizer → pre-training → base model → post-training → assistant → inference in an app
```

## Key definitions

| Term | Meaning |
|---|---|
| Token | A piece of text (byte, subword, word, punctuation) with an integer id |
| Next-token prediction | The model outputs a probability for every vocabulary token; generation repeats one token at a time |
| Base model | Output of pre-training; continues text |
| Assistant | Base model after post-training (instructions, preferences); answers and follows formats |
| WARC / WAT / WET | Common Crawl formats: raw responses / metadata / extracted plain text |
| Shingles | Word n-grams of a document (here n = 5) |
| Jaccard similarity | \|A ∩ B\| / \|A ∪ B\| of two shingle sets |
| MinHash | Signature whose matching positions estimate Jaccard similarity |
| LSH banding | Compare only documents sharing a whole band of the signature |
| Decontamination | Removing training text that overlaps evaluation sets |
| BPE | Start from 256 bytes; repeatedly merge the most frequent adjacent pair |
| Embedding | Learned vector for each token id |
| Q, K, V | Query (what I look for), key (what I offer), value (what I hand over) |
| Causal mask | Position i may only attend to positions j ≤ i |
| RoPE | Rotates Q and K by position so scores depend on relative distance |
| MoE | Router sends each token to a few experts; total vs active parameters |
| GQA | Several query heads share one key/value head; smaller KV cache |

## Formulas

| What | Formula |
|---|---|
| Precision / recall | TP / (TP + FP) · TP / (TP + FN) |
| LSH 50% threshold | (1 / bands) ^ (1 / rows); 16 bands × 8 rows ≈ 0.707 |
| Attention | softmax(Q Kᵀ / √d_head + mask) V |
| Cross-entropy | − ln p(correct next token); uniform guess = ln V (ln 1024 = 6.931) |
| Parameters per layer (classic) | attention 4D² + feed-forward 2DF; non-embedding ≈ 12 L D² when F = 4D |
| Embeddings | V × D (plus V × D again if the output layer is not tied) |
| Weight memory | parameters × bytes (fp32 4, bf16 2, int8 1, int4 0.5) |
| Training compute | ≈ 6 × parameters × tokens FLOPs |
| KV cache | 2 × layers × kv_heads × d_head × bytes, per token |
| MoE | total = shared part + all experts; active = shared part + chosen experts |

## Module numbers to remember (from the lab)

| Result | Value |
|---|---|
| Crawl → kept pages | 134 → 82 (labels: 83) |
| Pipeline precision / recall | 1.000 / 0.988; with `--min-words 50`: recall 0.687 |
| HTML characters kept as text | 25.4% |
| Near-duplicates vs different articles (Jaccard) | about 0.97 vs at most 0.417 |
| Tokens per word, en / es (vocab 1024) | 2.932 / 3.345 (ratio 1.141); 1.080 with Spanish ×5 |
| Neural bigram validation loss | 6.932 → 2.985 (counting bigram 3.065) |
| `classic-small` parameters | 124,439,808 (embeddings 31.0%) |
| `moe-8x-top2` total / active | 4,728,358,912 / 1,406,470,144 |

## Decision tables

| If you need to... | Prefer |
|---|---|
| Catch identical copies cheaply | URL normalisation + exact text hash |
| Catch edited copies at scale | MinHash + LSH |
| Keep useful pages with personal data | Redact, do not drop |
| Shorter sequences in a language | More training data in that language for the tokenizer, or a bigger vocabulary |
| Plan memory for an MoE model | Total parameters + KV cache |
| Plan latency for an MoE model | Active parameters |
| Keep data on your machines | Open weights, self-hosted |

## Lab commands

```bash
uv run lab ngram          # next-word probabilities, backoff, n-gram coverage
uv run lab ask            # base-style continuation vs assistant (LLM_PROVIDER)
uv run lab clean          # cleaning pipeline + precision/recall (--min-words, --skip)
uv run lab minhash        # exact vs MinHash Jaccard
uv run lab bpe            # train BPE, tokens per word en/es (--vocab-size, --es-weight, --text)
uv run lab attention      # shapes + causal attention weights
uv run lab train          # gradient descent (--lr, --steps)
uv run lab params         # parameters, memory, KV cache (--show)
uv run lab facts          # published figures with sources
```

## Three most common mistakes

1. **Trusting fluent output as fact.** Next-token prediction produces likely text; ground and verify.
2. **Copying a filter threshold without measuring it.** A 50-word minimum cut recall from 0.988 to
   0.687 on this crawl.
3. **Masking attention after the softmax.** Mask the scores with minus infinity before the softmax.

---

[← Previous: Quiz](quiz.md) · [🏠 Module 01](README.md) · [Next: Module 02 →](../02-decoding-post-training-evaluation/README.md)
