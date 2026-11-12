🌐 English · [Español](quiz.es.md)

# Module 01 quiz — How LLMs are built

Answer each question before opening the solution. Write your answers down; explaining *why* counts as
much as the letter. Questions marked 🔁 revisit the first lesson of this module.

### Q1

At each step of generation, what does a large language model produce?

- A. The next full sentence
- B. A probability for every token in its vocabulary
- C. The most relevant document from its training data
- D. A single word chosen from a dictionary

<details><summary>Answer</summary>

**B.** The model outputs a distribution over its whole vocabulary; one token is chosen (greedily or
by sampling), appended, and the loop repeats. It does not retrieve documents (C), and tokens are
often pieces of words, not dictionary words (D).

</details>

### Q2

Which Common Crawl format contains plain text that has already been extracted from the pages?

- A. WARC
- B. WAT
- C. WET
- D. JSONL

<details><summary>Answer</summary>

**C. WET.** WARC holds the raw HTTP responses and WAT holds metadata such as links. Quality-focused
pipelines often start from WARC and do their own extraction, because generic extracted text keeps
boilerplate.

</details>

### Q3

The lab's pipeline reports `TP=82 FP=0 FN=1 TN=51`, with "keep" as the positive class. Compute
precision and recall.

<details><summary>Answer</summary>

Precision = TP / (TP + FP) = 82 / 82 = **1.000**. Recall = TP / (TP + FN) = 82 / 83 = **0.988**.
Every kept page deserved to be kept; one page that deserved to be kept (the Corvid Thermo X product
page) was dropped.

</details>

### Q4

Your repetition filter drops short product pages that repeat the product name. What would you do?

<details><summary>Answer</summary>

Look at the failing pages first (the lab lists them). Then change the rule narrowly: apply the
repetition check only above a minimum length, or ignore repeats of the title. Rerun and check that
recall improves without letting spam through (precision). Finally, confirm on a labelled sample you
did not tune on, because tuning and scoring on the same pages flatters the result.

</details>

### Q5

Spot the bug. This MinHash helper gives different signatures every time the program starts, so
yesterday's index cannot be compared with today's.

```python
def shingle_ids(items: set[str]) -> list[int]:
    return [hash(s) % PRIME for s in items]
```

<details><summary>Answer</summary>

Python's built-in `hash()` for strings is salted with a random seed per process, so the same shingle
gets a different integer in every run. Use a stable hash such as `hashlib.blake2b` or SHA-256, as the
lab's `_shingle_ids` does. Sorting the set before hashing also keeps the output order deterministic.

</details>

### Q6

A byte-level BPE tokenizer is trained on the text `ab ab ab ba`. Pre-tokenization produces the
chunks `ab`, ` ab`, ` ab` and ` ba`. Which pair is merged first, and how many times does it occur?

<details><summary>Answer</summary>

`a` + `b`, which occurs **3 times** (once in `ab`, twice in ` ab`). The pair space + `a` occurs 2
times; space + `b` and `b` + `a` once each. BPE merges the most frequent pair first.

</details>

### Q7

Why does a byte-level BPE tokenizer never need an "unknown" token?

- A. It is trained on every language in the world
- B. It replaces rare words with their most similar known word
- C. All 256 byte values are in the base vocabulary, so any UTF-8 text can be encoded
- D. It drops characters it does not recognise

<details><summary>Answer</summary>

**C.** Any text is a sequence of bytes, and every byte is a token. Rare characters cost more tokens
(an accented letter may become two byte tokens), but nothing is lost: `decode(encode(text)) == text`.

</details>

### Q8

Spot the bug in this causal attention:

```python
scores = q @ k.T / math.sqrt(d)
weights = softmax(scores, axis=-1)
weights = np.where(causal_mask(t), weights, 0.0)
out = weights @ v
```

<details><summary>Answer</summary>

The mask is applied after the softmax. Each row was normalised using future scores, so earlier
outputs still depend on future tokens and rows no longer sum to 1. Mask the **scores** with minus
infinity before the softmax: `scores = np.where(causal_mask(t), scores, -np.inf)`.

</details>

### Q9

Your model has a 1,024-token vocabulary. Before any training, its next-token loss should be close to
which value, and what would you suspect if it started several times higher?

<details><summary>Answer</summary>

About ln(1024) = **6.931** nats, the loss of a uniform guess (the lab measures 6.932 and 6.905).
A much higher starting loss means the initial logits are far from equal: weights initialised too
large, a missing normalisation, or a bug such as unshifted targets or wrong indexing.

</details>

### Q10

Use Kaplan's rule of thumb to estimate the non-embedding parameters of a classic decoder-only model
with 8 layers and width 512 (feed-forward width 4 × 512).

<details><summary>Answer</summary>

12 × L × D² = 12 × 8 × 512² = **25,165,824**, about 25 million. The rule counts attention (4D² per
layer) and the feed-forward network (8D² per layer) and ignores biases and norms; for the lab's
`classic-small` it lands within 0.2% of the exact count.

</details>

### Q11

A mixture-of-experts model has 8 experts of 100 million parameters each, routes each token to 2 of
them, and has 200 million parameters outside the experts. What are its total and active parameter
counts?

<details><summary>Answer</summary>

Total = 200M + 8 × 100M = **1,000 million**. Active = 200M + 2 × 100M = **400 million**. Memory must
hold all 1,000 million; compute per token follows the 400 million.

</details>

### Q12 🔁 (lesson 1.1)

Every page a model learned from was correct. Explain how it can still state a wrong fact, using the
lab's Ember T3 example.

<details><summary>Answer</summary>

The model produces likely continuations, not checked facts. After "on 2023 -", the counting
model had seen two dates equally often (the Ember T3 and the Corvid Thermo X), and its three-word
context no longer contained the product name, so it produced 2023-05-22 instead of 2023-10-02.
Grounding in documents and verification are needed when facts matter.

</details>

### Q13 🔁 (lesson 1.1)

With a fixed training compute budget, what did the Chinchilla paper recommend?

- A. Train the largest model that fits in memory on whatever data is available
- B. Grow model size and the number of training tokens together
- C. Train a small model for as long as possible
- D. Spend the budget on post-training instead

<details><summary>Answer</summary>

**B.** For compute-optimal training, parameters and training tokens should be scaled in equal
proportion. Its 70-billion-parameter model, trained on more data with the same compute as the
280-billion-parameter Gopher, outperformed it.

</details>

### Q14

Two candidate models report the same score on a public benchmark. Rio asks which one to use for the
Hearthwise support chatbot. What do you do?

<details><summary>Answer</summary>

Treat equal scores as a reason to test both, not a tie-break. Fill the comparison sheet: total and
active parameters and memory, context length and KV cache, tokens per word on real English and
Spanish tickets, licence and data-residency constraints. Then run both on Hearthwise's own evaluation
set (module 02), and check the benchmark for possible contamination or different prompt settings.

</details>

### Q15

Grouped-query attention (GQA), as used in the Qwen2.5 and Gemma 3 families, mainly reduces:

- A. The parameters of the feed-forward layers
- B. The memory of the KV cache during generation
- C. The size of the vocabulary
- D. The number of training tokens needed

<details><summary>Answer</summary>

**B.** Several query heads share one key/value head, so fewer keys and values are cached per token.
In the lab, `modern-1b` with 4 key/value heads for 16 query heads needs a quarter of the cache it
would need without sharing.

</details>

## Scoring guide

| Correct answers | What it means |
|---|---|
| 13–15 | You are ready for module 02. |
| 10–12 | Good. Reread the self-check of the lessons behind the questions you missed. |
| 7–9 | Redo the related lab steps and the Break-it items before moving on. |
| 0–6 | Go back through the lessons in order; the lab outputs are the best anchor. |

## ⏭️ Next

[Module 02 — From base model to assistant →](../02-decoding-post-training-evaluation/README.md)

---

[← Previous: Lab](lab/README.md) · [🏠 Module 01](README.md) · [Next: Cheatsheet →](cheatsheet.md)
