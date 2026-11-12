🌐 English · [Español](cheatsheet.es.md)

# Module 05 cheatsheet — RAG I: Retrieval

## Definitions

| Term | Meaning |
|---|---|
| RAG (retrieval-augmented generation) | Search trusted documents, put the best passages in the prompt, answer from them with citations |
| Indexing pipeline (offline) | Parse → filter and de-duplicate → chunk → index (keyword, vector, facts) |
| Query pipeline (online) | Question → search → rerank → build prompt → generate |
| Chunk | The unit you index, retrieve and cite (`kb-007#4` = document `kb-007`, piece 4) |
| Inverted index | term → postings {chunk: term frequency} |
| Embedding / bi-encoder | Text → vector; chunks embedded offline, questions online, same model |
| Knowledge-based index | Facts as (subject, predicate, object) triples with a source document |
| ANN (approximate nearest neighbour) | Trades a little recall for much less work per query (IVF, HNSW, PQ) |
| Reranking | Reorder a small candidate set with a slower, stronger scorer (a cross-encoder in production) |

## Formulas

```text
IDF(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
BM25(q, d)  = sum over t in q of IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
              k1 = 1.5 caps repetition, b = 0.75 normalises length
cosine(u,v) = u . v / (|u| |v|)        (= dot product for unit vectors)
RRF(item)   = sum over lists of 1 / (60 + rank)
recall@k    = share of questions with a relevant document in the top k
MRR         = mean of 1 / rank of the first relevant document (0 if none)
recall@10 (ANN) = share of the exact top 10 that the approximate index returns
```

## Decision tables

| Need | Choose |
|---|---|
| Behaviour, tone, format | Prompting or fine-tuning (module 04) |
| Small, static, public corpus | Try long context first; keep it as a baseline |
| Changing facts, citations, per-user access | RAG |

| Source | Parser |
|---|---|
| Markdown, HTML, JSON | Rules (front matter, meta tags, headings, tables); drop nav, footer, hidden elements |
| Digital PDF | Layout model + text extraction; validate counts |
| Scans, image tables | OCR or a vision-language model; validate and keep page ids |

| Chunker | Use when | Watch out for |
|---|---|---|
| Fixed + overlap | You need a quick baseline | Cuts tables and sentences |
| Recursive | Prose with paragraphs | Ignores headings |
| Heading | Structured help articles | Big sections and tables become one long chunk |
| Semantic | Long prose with topic shifts | Only as good as the embedding |
| Table-row | Wide tables | Many small chunks from one page |

| Collection | Index |
|---|---|
| Small (Hearthwise today: 128 chunks) | Flat, exact |
| Large, memory available | HNSW, tune `efSearch` |
| Large, memory tight | IVF, tune `nprobe` |
| Too big for memory | IVF-PQ + exact rerank of candidates |

## Numbers to remember (from the lab)

| Result | Value |
|---|---|
| Unfiltered: questions (of 40) with a superseded or internal doc in the top 5 | 15 → 0 with filters |
| Heading chunks, `table` recall@5 | 0.00 → 1.00 with table-row chunks |
| BM25 / hashing vectors / hybrid recall@5 (table-row chunks) | 0.944 / 0.861 / 0.917 |
| Lexical reranker: recall@1 and MRR | 0.667 → 0.833 and 0.786 → 0.898 |
| One question out of 36 | about 0.028 of recall |
| IVF nprobe 1 / 8 / 32: recall@10 | 0.741 / 0.917 / 0.977 |
| HNSW efSearch 16 / 64 / 128: recall@10 | 0.819 / 0.951 / 0.981 |
| Memory: flat / HNSW / IVF-PQ (50,000 × 64) | 12.80 / 26.41 / 1.33 MB |

## Lab commands

```bash
uv run lab parse                                  # structure, metadata, hidden elements
uv run lab chunk --chunker table --show kb-007    # filters, duplicates, chunks of one doc
uv run lab index                                  # sizes; timings are examples
uv run lab search "question" [--fold] [--rerank] [--retriever hybrid] [--no-filter]
uv run lab prompt "question"                      # what a generator would receive
uv run lab eval                                   # the lineup; add a flag for one config by category
uv run lab kg "Which Hearthwise device was launched most recently?"
uv run lab ann [--no-faiss] [--spread 3]          # exact vs IVF vs HNSW vs PQ
```

## Three most common mistakes

1. **Filtering after retrieval.** Superseded and internal documents must never reach the index or the
   prompt; a passage in the prompt can be quoted.
2. **Trusting a metric without reading the chunks.** Fixed-size chunks "find" the table but lose its
   headers; doc recall cannot see that.
3. **Tuning on the evaluation set and reporting it as a result.** A perfect score on 36 questions you
   tuned on is a reason to write new questions.
