🌐 English · [Español](quiz.es.md)

# Module 05 quiz — RAG I: Retrieval

Thirteen questions. Answer each one on paper before opening the answer. Questions marked 🔁 review
earlier modules.

### Q1

Which of these problems does retrieval-augmented generation (RAG) **not** solve on its own?

- A. The model has never heard of a product launched after its training data was collected.
- B. The returns policy changed last month.
- C. The model's answers are too long and too formal.
- D. Customers want to see which article an answer came from.

<details><summary>Answer</summary>

**C.** Tone and length are behaviour; you shape them with prompting or fine-tuning. RAG fixes knowledge
cut-off (A), changing facts (B) and citations (D) by putting chosen, current passages in the prompt.

</details>

### Q2

Without filters, BM25 ranks the superseded 2023 returns policy (`kb-010`, "14 days") first for "What is
the return window for Hearthwise products?". What is the best fix?

- A. Add "always prefer the newest policy" to the system prompt.
- B. Drop documents whose `status` is not `current` before indexing.
- C. Retrieve 20 chunks instead of 5 so the current policy is also included.
- D. Switch from BM25 to embeddings.

<details><summary>Answer</summary>

**B.** The stale document should never reach the index. A prompt rule (A) asks the model to resolve a
conflict you could have removed; more chunks (C) keep the wrong one in the context; embeddings (D) would
also find the old policy, because it is about the same topic.

</details>

### Q3

The word "watcher" appears in 4 of the 111 heading-level chunks. Compute its BM25 inverse document
frequency, `ln(1 + (N - df + 0.5) / (df + 0.5))`, to two decimals.

<details><summary>Answer</summary>

`ln(1 + (111 - 4 + 0.5) / (4 + 0.5)) = ln(1 + 107.5 / 4.5) = ln(24.89) = 3.21`. For comparison, "the"
(in 75 chunks) has an IDF of 0.39: rare terms carry the score.

</details>

### Q4

Spot the bug. Recall@5 dropped from 0.944 to 0.861 after this change to `BM25.scores`:

```python
def scores(self, query: str) -> list[float]:
    out = [0.0] * self.index.n_docs
    for term in query.split():
        ...
```

<details><summary>Answer</summary>

The index was built with `tokenize` (lower-case, punctuation removed) but the query is split on spaces.
"Watcher" no longer matches "watcher" and "weigh?" matches nothing. Documents and queries must use the
same tokenizer: `for term in tokenize(query, fold=self.fold):`.

</details>

### Q5

Reciprocal rank fusion with `k = 60`. Chunk X is ranked 2nd by BM25 and does not appear in the vector
list. Chunk Y is ranked 10th in both lists. Which one ranks higher after fusion?

<details><summary>Answer</summary>

**Y.** X scores 1 / 62 = 0.0161. Y scores 1 / 70 + 1 / 70 = 0.0286. RRF rewards agreement, which is why
the Spanish shipping chunk fell from 2nd (BM25) to 31st in the lab's hybrid search.

</details>

### Q6

Which chunker writes every table value next to its column name?

- A. Fixed-size with overlap
- B. Recursive
- C. Semantic
- D. Table-row

<details><summary>Answer</summary>

**D.** Each row of a wide table becomes a chunk such as `Compare Hearthwise devices | Watcher S1 | ...;
Weight (g): 34; ...`. Fixed-size and recursive chunks also found the table document in the lab, but their
chunks contain "... yes 34 -10 to 45 ..." with the header row in another chunk.

</details>

### Q7

A teammate sees `bm25/fixed` at recall@5 of 0.972 and 1.00 on the `table` category, above `bm25/table`
(0.944), and wants to switch. What do you check before agreeing?

<details><summary>Answer</summary>

Read the chunks the generator would receive. The fixed-size chunk with the Watcher S1 row has no column
names, so a model must guess what "34" means. Also check the size of the difference: one question is
worth about 0.028 on 36 questions, so 0.972 against 0.944 is a single question. A fair decision weighs
recall against chunk readability, and ideally runs module 06's answer evaluation on both.

</details>

### Q8

In the lab's benchmark, the HNSW index takes 26.41 MB while the raw vectors take 12.80 MB. Why?

- A. HNSW stores every vector twice.
- B. HNSW stores a graph: up to 2 × M neighbour links per vector on its bottom layer, plus upper layers.
- C. FAISS pads HNSW vectors to 128 dimensions.
- D. HNSW keeps the k-means centroids of every layer.

<details><summary>Answer</summary>

**B.** With M = 32, each vector keeps up to 64 links of 4 bytes on the bottom layer, about as much as the
64-dimension vector itself. That is the price of HNSW's high recall at low work per query.

</details>

### Q9

IVF with 256 lists gives recall@10 of 0.741 at nprobe = 1, 0.917 at nprobe = 8 and 0.977 at nprobe = 32.
Sam's target is recall@10 of at least 0.97 with as little work per query as possible. Which setting do
you pick, and what does it cost compared with nprobe = 8?

<details><summary>Answer</summary>

nprobe = 32: the only IVF setting above 0.97. It computes 6,504 similarities per query against 1,814 at
nprobe = 8, about 3.6 times the work, while still far below the 50,000 of exact search. HNSW with
efSearch = 128 (0.981 with 3,055 similarities) is the alternative if the doubled memory is acceptable.

</details>

### Q10

The knowledge graph answers "What is the price of the Lookout C2 in Mexico?" with €89. The help centre
never states a price in Mexico. What would you do?

<details><summary>Answer</summary>

Route a question to the graph only when every entity and constraint in it maps to the schema. "In
Mexico" does not, so fall back to text retrieval, and let the generator say it does not know (q037 is
unanswerable on purpose). Never return a graph fact that answers a simpler question than the one asked.

</details>

### Q11 🔁 (module 02)

After many experiments on the 36 evaluation questions, one configuration reaches recall@5 of 1.000. In
the language of model evaluation, what went wrong, and what is the fix?

<details><summary>Answer</summary>

The evaluation set became a tuning set: choices were made after looking at its failures, so the score is
optimistic, the same problem as a benchmark that leaked into training. The fix is a fresh, held-out set
(new questions, ideally from real tickets) that nobody looked at while building the pipeline.

</details>

### Q12 🔁 (module 03)

In the LLM playground, long conversations eventually exceed the model's context window. Name two ways to
handle it, and explain why retrieved passages make the problem more pressing in a RAG chatbot.

<details><summary>Answer</summary>

Drop or truncate the oldest turns (a sliding window), or summarise earlier turns into a short memory.
In a RAG chatbot, the retrieved passages, the system rules and the conversation all share the same window,
so every chunk you retrieve leaves less room for history. Retrieving fewer, better chunks (reranking,
table-row chunks that carry their own context) is also a context-budget decision.

</details>

### Q13 🔁 (module 04)

Which statement about LoRA fine-tuning is true?

- A. It updates every weight of the model, but with a smaller learning rate.
- B. It freezes the model and trains small low-rank matrices added to some weight matrices.
- C. It needs no training data, only a system prompt.
- D. It is the best way to keep a model's knowledge of a changing returns policy up to date.

<details><summary>Answer</summary>

**B.** See the [LoRA paper](https://arxiv.org/abs/2106.09685). D is the trap: fine-tuning shapes
behaviour, and a fact learned into the weights stays there, stale, until you train again. Facts that
change belong in a retrieval index.

</details>

## 📊 Scoring guide

| Score | What it means | What to do |
|---|---|---|
| 11–13 | You can build and defend a retrieval pipeline | Move on to module 06 |
| 8–10 | The ideas are there; some details are shaky | Re-read the lessons behind the questions you missed and redo the matching lab step |
| 0–7 | The pipeline is not yet clear | Redo lessons 5.2 and 5.3 with the lab open, then retake the quiz |

## ⏭️ Next

[Module 06 — RAG II: The Support Chatbot](../06-rag-generation-and-evaluation/README.md): turn the retriever
into a grounded chatbot that cites, refuses when it should and resists injected instructions.
