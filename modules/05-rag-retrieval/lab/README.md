🌐 English · [Español](README.es.md)

# Lab 05 — Retrieval for the Hearthwise help centre

> Parse, filter, chunk and index the Hearthwise help centre, measure five chunkers and three
> retrievers on the shared evaluation questions, and benchmark exact against approximate search with
> FAISS. ⏱️ about 2 hours.

## 🎯 What you will build

The retrieval half of the support chatbot that module 06 completes:

| Piece | File | What it does |
|---|---|---|
| Parser | `src/m05_retrieval/parsing.py` | Markdown front matter and HTML (meta tags, headings, tables, hidden elements) into `Document` blocks |
| Chunkers and filters | `chunking.py` | Fixed, recursive, heading, semantic and table-row chunkers; metadata filters; near-duplicate removal |
| Keyword index | `bm25.py` | Inverted index and BM25 scoring |
| Vector index | `vectors.py` | Offline hashing embedder (or a real one through `providers.py`) and exact cosine search |
| Knowledge index | `knowledge.py` | Product facts as (subject, predicate, object) triples with their source |
| Fusion and reranking | `hybrid.py`, `rerank.py` | Reciprocal rank fusion and a lexical second-stage reranker |
| Pipeline and evaluation | `retrieval.py`, `evaluate.py` | One `Config` builds the whole pipeline; recall@k, MRR, ctx@5 and exposure on `rag_eval` |
| ANN benchmark | `ann.py` | Exact numpy k-NN, a numpy IVF and FAISS Flat, IVF, HNSW and IVF-PQ on seeded vectors |
| Prompt | `prompting.py` | The grounded prompt a generator would receive |

Everything runs offline on a CPU, with no API key. The same command prints the same numbers every time
(only the timings change).

## ⚙️ Setup

From the repository root:

```bash
cd modules/05-rag-retrieval/lab
uv sync
uv run pytest
```

Expected summary (the time varies by machine):

```text
118 passed, 1 skipped in 6.70s
```

The skipped test talks to a real Ollama server; see [🔌 Using a real model](#-using-a-real-model).
`uv run lab --help` lists the commands: `parse`, `chunk`, `index`, `search`, `eval`, `prompt`, `kg`
and `ann`.

## 👣 Steps

Every step has a command, its expected output and a **✍️ Build it** task. The reference code is
already in place so that everything runs. To practise, delete the body of the named function (keep
its signature and docstring), write your own version, and run the listed test. `git restore <file>`
brings the reference back if you get stuck.

### Step 1 — Parse the help centre

```bash
uv run lab parse
```

The first and last lines of the output:

```text
doc     format   lang status     audience headings texts tables hidden
kb-001  markdown en   current    public          6    13      1      0
kb-002  markdown en   current    public          6    11      1      0
...
kb-040  markdown en   current    public         13    34      2      0
kb-041  markdown en   current    internal        4     3      1      0

26 documents, 13 tables (80 data rows), 1 hidden HTML element(s) dropped
about 5,070 tokens of parsed text (rough count: words and punctuation)
```

Find `kb-010` (superseded) and `kb-041` (internal) in the table, and `kb-025`, the page with one
hidden element.

**✍️ Build it:** rewrite `split_front_matter` and `parse_markdown_body` in `parsing.py`. Check with
`uv run pytest tests/test_parsing.py`.

### Step 2 — Filter and chunk

```bash
uv run lab chunk --chunker table --show kb-007
```

```text
near-duplicate: kb-027 ~ kb-026 (Jaccard 0.884)
dropped kb-010: status=superseded
dropped kb-027: near-duplicate of kb-026 (Jaccard 0.884)
dropped kb-041: audience=internal

chunker=table: 23 documents -> 128 chunks (words per chunk: mean 31.1, max 96)
wrote outputs/chunks-table.jsonl

--- kb-007#0 [Compare Hearthwise devices]
Compare Hearthwise devices
Use this table to compare the main specifications of every device we sell. Prices include VAT for Spain and Portugal.
Standby power is measured with the device connected to Wi-Fi and idle.

--- kb-007#1 [Compare Hearthwise devices]
Compare Hearthwise devices | Ember T3 | Price (EUR): 149; Wi-Fi: 2.4 GHz; Standby (W): 0.6; Needs hub: no; Weight (g): 210; Operating temperature (°C): 5 to 35; Launched: 2023-10-02
```

The output continues with one chunk per device. Repeat with `--chunker heading`, `fixed`, `recursive`
and `semantic`, and open `outputs/chunks-<name>.jsonl` to compare them.

**✍️ Build it:** rewrite `chunk_fixed` (windows with overlap) and `row_to_text` in `chunking.py`. Check
with `uv run pytest tests/test_chunking.py`.

### Step 3 — Build the indexes

```bash
uv run lab index
```

```text
chunks:              128
chunk text:          22,048 bytes
vocabulary:          950 terms
postings:            3,361 (term, chunk) pairs
inverted index:      ~32,064 bytes (8 bytes per posting + the terms)
vector matrix:       131,072 bytes (128 x 256 float32)

build time (example, varies by machine): bm25 1.1 ms, vector 12.4 ms
query time (example, varies by machine): bm25 0.04 ms, vector 0.08 ms
```

**✍️ Build it:** rewrite `InvertedIndex.build`, `idf` and `BM25.term_score` in `bm25.py`. Check with
`uv run pytest tests/test_bm25.py`.

### Step 4 — Search and see the prompt

```bash
uv run lab search "Who pays customs fees for orders shipped to Mexico?" --fold
```

```text
config: bm25/table/fold  (128 chunks)
1. kb-013#1  score=6.493  Spain and Portugal - Standard delivery takes 2 to 4 business days and costs €4.95. - Stand
2. kb-031#0  score=4.867  Envíos a Latinoamérica Enviamos a México, Colombia y Chile con una tarifa plana de **€14.9
3. kb-031#1  score=4.468  Plazos de entrega País Plazo México 7 a 10 días hábiles Colombia 8 a 12 días hábiles Chile
4. kb-026#3  score=4.069  Where can I download an invoice? In the app under Account > Orders. Tap the order and choo
5. kb-011#2  score=3.762  How to start a return 1. Open the Hearthwise app and go to Account > Orders. 2. Choose the
query time: 0.11 ms (example, varies by machine)
```

Now fuse BM25 with the hashing vectors:

```bash
uv run lab search "Who pays customs fees for orders shipped to Mexico?" --fold --retriever hybrid
```

```text
config: hybrid/table/fold  (128 chunks)
1. kb-026#3  score=0.032  Where can I download an invoice? In the app under Account > Orders. Tap the order and choo
2. kb-013#4  score=0.032  Tracking your order You get a tracking link by email when the parcel leaves our warehouse.
3. kb-013#1  score=0.030  Spain and Portugal - Standard delivery takes 2 to 4 business days and costs €4.95. - Stand
4. kb-003#4  score=0.029  Resetting to factory settings Hold the side button for 10 seconds until the LED blinks red
5. kb-006#4  score=0.029  Resetting to factory settings Hold the button on the back for 20 seconds until the ring li
query time: 0.20 ms (example, varies by machine)
```

The Spanish page is gone: lesson 5.3 explains why. Finally, look at what a generator would receive:
`uv run lab prompt "Can I return an opened Glow A19?"` (output in lesson 5.1).

**✍️ Build it:** rewrite `rrf_fuse` in `hybrid.py` and `format_context` in `prompting.py`. Check with
`uv run pytest tests/test_vectors_hybrid_rerank.py tests/test_prompting_and_providers.py`.

### Step 5 — Evaluate chunkers and retrievers

```bash
uv run lab eval
```

```text
36 answerable questions. R@k, MRR: document ranking. ctx@5: first 5 chunks.
table, x-ling: R@5 per category. exposed: questions (of 40) with a superseded or internal doc in the top 5.

config                       chunks   R@1   R@5   MRR ctx@5 table x-ling exposed
bm25/heading/nofilter           119 0.611 0.917 0.745 0.917  0.00   0.50      15
bm25/heading                    111 0.667 0.889 0.770 0.861  0.00   0.50       0
bm25/fixed                       65 0.750 0.972 0.840 0.944  1.00   0.50       0
bm25/recursive                   52 0.778 0.972 0.863 0.972  1.00   0.50       0
bm25/semantic                    98 0.639 0.917 0.768 0.917  0.00   0.50       0
bm25/table                      128 0.667 0.944 0.786 0.917  1.00   0.50       0
vector/table                    128 0.722 0.861 0.804 0.861  0.00   0.00       0
hybrid/table                    128 0.750 0.917 0.816 0.889  0.50   0.00       0
bm25/table/fold                 128 0.667 0.972 0.800 0.944  1.00   1.00       0
bm25/table/rerank               128 0.833 0.972 0.898 0.944  1.00   0.50       0
bm25/table/fold/rerank          128 0.833 1.000 0.905 0.972  1.00   1.00       0
```

The first row matches the reference baseline in [`datasets/README.md`](../../../datasets/README.md)
for recall@5 (0.917) and the `table` and `cross_lingual` categories (0.00 and 0.50); the lab splits HTML
pages by heading, so it has 119 chunks instead of the baseline's 118 and an MRR of 0.745 instead of
0.754. Any flag (`--chunker`, `--retriever`, `--no-filter`, `--fold`, `--rerank`, `--title`) switches
to a single configuration with recall@5 per category and the list of missed questions.

**✍️ Build it:** rewrite `doc_ranking`, `recall_at_k` and `reciprocal_rank` in `evaluate.py`. Check
with `uv run pytest tests/test_evaluate.py tests/test_documented_numbers.py`.

### Step 6 — Ask the knowledge graph

```bash
uv run lab kg "Which Hearthwise device was launched most recently?"
```

```text
90 triples about 6 products
(Lookout C2, Launched, 2025-09-01)  source: kb-005
```

Then ask `uv run lab kg "What is the price of the Lookout C2 in Mexico?"`. It answers €89 from
`kb-005` and 89 from `kb-007`, although the help centre never states a price in Mexico.

**✍️ Build it:** `uv run lab kg "What is the SKU of the Watcher S1?"` finds nothing, although every
product page lists its SKU. Add a predicate hint to `PREDICATE_HINTS` in `knowledge.py` so that it
returns `(Watcher S1, SKU, HW-WCH-S1)`, and add a test for it in `tests/test_knowledge.py`.

### Step 7 — Exact vs approximate search

```bash
uv run lab ann
```

```text
50,000 vectors x 64 dims, 200 queries, recall@10 against exact search

method               params                    recall@10  compared memory MB ms/query*
numpy exact          -                             1.000    50,000     12.80     0.472
numpy IVF            nlist=256 nprobe=1            0.750       484     13.27     0.035
numpy IVF            nlist=256 nprobe=8            0.917     1,825     13.27     0.179
faiss IndexFlatIP    -                             1.000    50,000     12.80     0.233
faiss IndexIVFFlat   nlist=256 nprobe=1            0.741       474     13.27     0.004
faiss IndexIVFFlat   nlist=256 nprobe=8            0.917     1,814     13.27     0.025
faiss IndexIVFFlat   nlist=256 nprobe=32           0.977     6,504     13.27     0.048
faiss IndexHNSWFlat  M=32 efSearch=16              0.819       649     26.41     0.015
faiss IndexHNSWFlat  M=32 efSearch=64              0.951     1,681     26.41     0.053
faiss IndexHNSWFlat  M=32 efSearch=128             0.981     3,055     26.41     0.112
faiss IndexIVFPQ     nlist=256 m=16 nprobe=8       0.593     1,814      1.33     0.017

* example timings, vary by machine. compared = similarities computed per query
```

It takes a few seconds. On another CPU, recall can differ in the last digit; the tests allow ±0.02.

**✍️ Build it:** rewrite `exact_knn` and `NumpyIVF.search` in `ann.py`. Check with
`uv run pytest tests/test_ann.py`, then run `uv run lab ann --no-faiss` and compare your IVF with
FAISS's.

## 💥 Break it

**1. Switch the filters off.** Run `uv run lab eval --chunker table --no-filter`. What happens to
recall@5, MRR and the `exposed` column?

<details><summary>What you should see</summary>

Recall@5 stays at 0.944, MRR drops from 0.786 to 0.758 and `exposed` jumps from 0 to 15. Recall alone
would tell you nothing is wrong. The old policy and the internal playbook are now competing for the top
5, and for q039 the internal goodwill section is the very first result.

</details>

**2. Remove length normalisation.** In `bm25.py`, change the default `b: float = 0.75` to `b: float = 0.0`
and run `uv run lab eval --chunker heading`.

<details><summary>What you should see</summary>

Recall@5 rises from 0.889 to 0.944, MRR from 0.770 to 0.843, and the `table` category goes from 0.00 to
1.00: the long comparison-table chunk is no longer penalised. But q021 (delivery to Chile) is now missed.
`b` is a knob, not a law; turning it changes which questions win. Restore `0.75` afterwards.

</details>

**3. Tokenise queries differently from documents.** In `BM25.scores`, replace
`tokenize(query, fold=self.fold)` with `query.split()` and run `uv run lab eval --chunker table`.

<details><summary>What you should see</summary>

Recall@5 falls from 0.944 to 0.861, MRR from 0.786 to 0.739, and 5 questions are missed: "Watcher" no
longer matches "watcher", and "weigh?" (with its question mark) matches nothing. Documents and queries
must go through the same tokenizer.

</details>

**4. Destroy the clusters.** Run `uv run lab ann --no-faiss --spread 3`. The vectors are now much less
clustered. What happens to the numpy IVF?

<details><summary>What you should see</summary>

Recall@10 collapses to 0.069 at nprobe = 1 and 0.301 at nprobe = 8 (against 0.750 and 0.917 with the
default spread). IVF assumes that near neighbours share a cluster; when the data has no cluster
structure, probing a few lists finds few neighbours. Real embeddings are clustered, but how much is
something you measure, not assume.

</details>

## 🚀 Extend it

**1. Contextual chunk headers.** Run `uv run lab eval --chunker table --title`, which prefixes every
chunk with its document title. Then extend `with_title` in `chunking.py` to add the section heading
too (`Title > Section`) and measure again.

<details><summary>Hint and reference result</summary>

With the title prefix, recall@1 goes from 0.667 to 0.722, recall@5 from 0.944 to 0.972 and MRR from
0.786 to 0.824; only q022 is missed. The title brings words like "Returns policy" or "Factory reset" into
every chunk of its page, which is exactly the context a heading-level chunk loses. For the section, use
`Chunk.section`, which every chunker already fills.

</details>

**2. Bilingual query expansion.** Make q022 work without accent folding: add a small English to Spanish
glossary (customs, shipping, delivery, returns, warranty) and append the translations to the query
before searching.

<details><summary>Hint</summary>

Write `expand_query(q: str) -> str` in a new `query.py`, apply it in `BM25Retriever.search`, and expose
it as a `Config` flag so `uv run lab eval` can compare it. Check the `cross_lingual` column and the
`exposed` column: expansion must not drag in unrelated pages. A real system would translate the query
with an LLM or use a multilingual embedding model; the glossary shows the idea offline.

</details>

**3. Light stemming.** Teach the tokenizer that "weigh" and "weight", or "device" and "devices", are the
same term.

<details><summary>Hint</summary>

Add a `stem` option to `tokenize` in `text.py` that strips a few English suffixes (`-s`, `-es`, `-t`
after `weigh`, `-ing`), use it on both sides in `InvertedIndex.build` and `BM25.scores`, and measure.
Watch q004 (warranty "devices") and q017 (weigh). Then check that Spanish questions did not get worse.

</details>

**4. Compressed search with an exact second pass.** Wrap the IVF-PQ index in `faiss.IndexRefineFlat`
so that candidates found with compressed codes are re-scored with the full vectors.

<details><summary>Hint</summary>

In `_faiss_suite` in `ann.py`: `refine = faiss.IndexRefineFlat(ivfpq)`, then `refine.k_factor = 4` and
`refine.search(queries, k)`. Report recall@10 and memory (the refine index stores the raw vectors as
well, so memory goes up). Compare with plain IVF-Flat at the same `nprobe`.

</details>

**5. A real embedding model.** Run the vector and hybrid configurations with Ollama embeddings (see the
next section) and compare the `cross_lingual` and `table` columns with the hashing embedder.

<details><summary>Hint</summary>

`--embedder provider` is already wired into `search`, `eval` and `prompt`. Choose a multilingual
embedding model if you want the Spanish shipping page to match English questions, and record the model
name next to your numbers.

</details>

## ✅ Definition of Done

- [ ] `uv run pytest` passes (118 passed, 1 skipped).
- [ ] You rewrote at least four of the **Build it** functions and their tests pass.
- [ ] You can explain, with the chunk text, why `bm25/heading` misses q017 and q018 and `bm25/table`
      does not.
- [ ] You ran every **Break it** experiment and can say what each one teaches.
- [ ] You can explain why `hybrid/table` scores lower than `bm25/table` on this data.
- [ ] You can read the `ann` table and recommend an index for 128 chunks and for 50,000 vectors.
- [ ] You restored any code you changed for **Break it** (`git restore` or by hand).

## 🧯 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Could not find the repo's datasets/ folder` | The lab was copied outside the repository | Set `COURSE_DATASETS_DIR=/path/to/datasets` |
| `uv sync` cannot find a `faiss-cpu` wheel | Unusual platform or Python version | Use Python 3.12 (`uv python install 3.12`). If it still fails, remove `faiss-cpu` from `pyproject.toml`: the numpy parts of `uv run lab ann --no-faiss` still work and the FAISS rows of the tests are skipped |
| ANN recall differs from the table in the last digit | Different CPU instructions or FAISS version | Expected; the tests allow ±0.02 |
| `Network access is disabled in tests` | A test tried to reach a server | Tests are offline by design; mark live tests `ollama` or `openai` |
| `ProviderError: Cannot reach http://localhost:11434` | `--embedder provider` with `LLM_PROVIDER=ollama` and no server | Start Ollama (`ollama serve`) or drop `--embedder provider` |
| The numbers in `eval` changed | You edited a tokenizer, chunker or BM25 parameter | `git diff src/` shows what; `tests/test_documented_numbers.py` names the number that moved |

## 🔌 Using a real model

The lab's vectors come from the offline hashing embedder in `providers.py`, which matches spelling, not
meaning. To use a real embedding model, select a provider and pass `--embedder provider`; everything
else stays the same.

**Ollama (local, free):**

```bash
ollama pull nomic-embed-text              # the default embedding model in providers.py
LLM_PROVIDER=ollama OLLAMA_EMBED_MODEL=nomic-embed-text \
  uv run lab eval --retriever hybrid --chunker table --embedder provider
```

**Any OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp server, a hosted API):**

```bash
LLM_PROVIDER=openai OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_MODEL=<chat-model> \
  OPENAI_EMBED_MODEL=<embedding-model> OPENAI_API_KEY=<key-if-needed> \
  uv run lab eval --retriever vector --chunker table --embedder provider
```

`OPENAI_MODEL` is required by the provider even though only embeddings are used here.

**What was not tested.** These real-model paths were not run for this module: Ollama was installed on
the authoring machine but not running, and no API key was used. Every number in the lessons, this
README and the quiz comes from the offline hashing embedder. The live test
`tests/test_prompting_and_providers.py::test_ollama_embeddings_rank_the_returns_policy` runs only with
`RUN_LIVE_TESTS=1 uv run pytest -m ollama` and a running server. If you run it, write your numbers down
with the model name: they are yours, not the course's.

---

[← Previous: 5.4 Exact and approximate search](../04-exact-and-approximate-search.md) · [🏠 Module 05](../README.md) · [Next: Quiz →](../quiz.md)
