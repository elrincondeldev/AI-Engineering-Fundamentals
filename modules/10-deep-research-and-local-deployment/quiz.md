🌐 English · [Español](quiz.es.md)

# Module 10 quiz — Deep research and local deployment

14 questions. Answer each one before you open the solution. Questions marked 🔁 review earlier
modules.

### Q1

Which step of a deep-research pipeline should be plain code rather than a model call?

- A. Planning the sub-queries
- B. Writing the final report
- C. Checking that each cited number appears in the cited page
- D. Extracting evidence sentences from a page

<details><summary>Answer</summary>

**C.** Citation checking follows an exact rule and must not be persuadable by the text it checks.
Planning and writing need judgement (a reasoning model fits), and extraction is a narrow model task
whose output you then verify in code.

</details>

### Q2

The trace for one report says: 1 planning call, 20 pages opened by the reader, 1 writing call.
How many model calls did the report cost, and what share went to reading? Where would you look
first to cut cost?

<details><summary>Answer</summary>

22 calls, 20 of them for reading: about 91%. Cut cost where the calls are: use a smaller model for
the reader, open fewer pages per search, skip pages earlier with cheap checks (credibility,
duplicates), and cache pages that several sub-queries open.

</details>

### Q3 — Spot the bug

```python
quotes = [q.strip() for q in out.text.splitlines() if q.strip() and q.strip() != "NONE"]
words = " ".join(quotes).split()
passage = " ".join(words[:max_words])
```

<details><summary>Answer</summary>

The quotes are never verified against the page. A model that paraphrases or invents a sentence
("The Ember T3 costs €99") would create evidence that no source contains. The lab keeps only
`[q for q in quotes if q in text]` and counts the rejected ones, which also makes a misbehaving
model visible in the stats.

</details>

### Q4 — What would you do?

A teammate shows the ablation table: removing the sanitiser alone changes nothing (17 of 17 facts,
0 traps, 0 PII). They propose deleting the sanitiser to simplify the code. What do you answer?

<details><summary>Answer</summary>

Keep it. It changes nothing only because the credibility filter skips the forum before the reader
opens it. With both defences off, the injected instruction and the forum user's personal data reach
the report (2 traps, 2 PII leaks). Defence in depth means each layer is tested alone and in
combination, and kept because the others can fail.

</details>

### Q5

The same publisher measured the Corvid Thermo X at 0.9 W in 2024 and at 0.4 W in 2025 after a
firmware update. What should the report do?

- A. Average them: about 0.65 W
- B. Quote both without comment
- C. Use the 2025 figure and note that it replaces the 2024 one
- D. Use the 2024 figure because it was published first

<details><summary>Answer</summary>

**C.** The newer page explicitly replaces the older one. Averaging invents a number nobody measured,
and quoting both without comment leaves the reader to pick the wrong one. Recency weighting alone
does not remove the old page; the supersession rule (same publisher, same product, keep the newest)
does.

</details>

### Q6 — Calculation

How many GB do the weights of a 14-billion-parameter model need at 4 bits with one 16-bit scale per
group of 32 weights? (1 GB = 10^9 bytes.)

<details><summary>Answer</summary>

4 + 16/32 = 4.5 bits per weight. 14 x 10^9 x 4.5 / 8 = 7.875 x 10^9 bytes, about **7.88 GB**,
before the key-value cache and overhead.

</details>

### Q7 — Calculation

A model has 32 layers, 8 key-value heads and a head dimension of 128, and stores the cache in 16
bits. How large is the key-value cache for 4 conversations of 4,096 tokens each?

<details><summary>Answer</summary>

Per token: 2 x 32 x 8 x 128 x 2 bytes = 131,072 bytes (128 KiB). Times 4,096 tokens times 4
conversations: 2,147,483,648 bytes, about **2.15 GB**. The cache grows linearly with both context
and batch.

</details>

### Q8

In the lab, int4 quantisation with one scale for the whole tensor turns 99.6% of the weights into
zero. Why?

- A. 4 bits cannot represent negative numbers
- B. A few large outliers set the scale, so the steps are too coarse for the ordinary small weights
- C. Rounding always goes towards zero
- D. The scales are stored in 4 bits as well

<details><summary>Answer</summary>

**B.** The scale is max|w| / 7. With outliers ten times larger than normal weights, most weights are
smaller than half a step and round to 0. Per-channel and per-group scales give each block its own
range, which is why group-32 keeps the output error near 0.11 instead of 0.92.

</details>

### Q9

In the simulator, static batching gives a p50 time to first token of about 10 seconds and
continuous batching about 98 ms, on the same requests and step costs. What explains most of the
difference?

<details><summary>Answer</summary>

Queueing. With static batching a new request waits until the longest answer in the current batch
finishes; with continuous batching it joins at the next decode step. The time per output token is
similar for both (36.8 ms and 34.1 ms); the waiting is what differs.

</details>

### Q10 — Calculation

Speculative decoding with acceptance rate 0.8, 4 draft tokens per round and a draft step costing 5%
of a target step: how many tokens does each target pass produce on average, and what is the
expected speed-up? Use tokens = (1 - α^(k+1)) / (1 - α) and speed-up = tokens / (k x c + 1).

<details><summary>Answer</summary>

tokens = (1 - 0.8^5) / 0.2 = 3.362; speed-up = 3.362 / (4 x 0.05 + 1) = **2.80x**. The output
distribution is unchanged, because the target model checks every drafted token.

</details>

### Q11 — What would you do?

The cost model shows €1.80 a day for an API and €2.54 a day for a local machine at today's volume.
Jordan says: "So we use the API." What do you answer?

<details><summary>Answer</summary>

Not for this project: Noor's rule says prompts and market data must not leave the machine, so the
API is not an option at any price. The cost model tells Jordan what privacy costs (€0.74 a day here,
or only the €0.23 a day of energy if the laptop already exists) and when local becomes cheaper anyway
(1.4 times today's volume). Prices are placeholders; re-run with real ones.

</details>

### Q12 🔁 (module 05)

The lab's search is Okapi BM25. For which question would BM25 most likely fail to find the right
page in the snapshot?

- A. "smart thermostat sales 2025 growth Mexico" (the report uses these words)
- B. A question in English whose answer only appears on a Spanish-language page
- C. "Corvid Thermo X standby power"
- D. "household electricity price kWh Portugal"

<details><summary>Answer</summary>

**B.** BM25 scores shared terms. A Spanish page shares almost no words with an English query, so it
ranks low even when it contains the answer. Dense multilingual embeddings or query translation help
there.

</details>

### Q13 🔁 (module 08)

The lab's planner writes all sub-queries before any search, then reflects after each round. Which
earlier idea is closest to "plan first, then execute"?

- A. ReAct: interleave a thought and an action at every step
- B. ReWOO: plan the tool calls up front, then run them and combine the results
- C. Tree search over many alternative action sequences
- D. Self-consistency voting over sampled answers

<details><summary>Answer</summary>

**B.** Planning up front saves model calls and makes the cost predictable. The reflection round adds
a little of ReAct's adaptivity: if evidence is missing, the loop issues follow-up searches.

</details>

### Q14 🔁 (module 09)

Module 09 showed that extra inference-time compute (sampling several answers, verifiers) pays off on
problems with several dependent steps. Where in the research pipeline would you spend it?

<details><summary>Answer</summary>

On planning and synthesis: one call each per report, where a mistake affects everything downstream
(a missing aspect, a wrong comparison). Not on reading, which is called once per page and is
verified by code anyway. A verifier for synthesis can be cheap code: the citation check of the
grader is one.

</details>

## Scoring guide

| Score | What it means |
|---|---|
| 12–14 | Ready for the capstone. |
| 9–11 | Re-read the lessons behind the questions you missed and redo the matching lab step. |
| 0–8 | Work through the lab again, especially the ablations (step 5) and the memory estimator (step 6). |

## ⏭️ Next

You have finished the last module. Bring everything together in the
[capstone: the launch review of Hearthwise Assist](../../capstone/README.md).
