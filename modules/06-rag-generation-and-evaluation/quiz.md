🌐 English · [Español](quiz.es.md)

# Module 06 quiz — RAG II: grounded generation, RAFT and evaluation

Fourteen questions. Write your answer before you open the solution. Questions marked 🔁 review
earlier modules.

### Q1 · Multiple choice

The help centre keeps a superseded 2023 returns policy next to the current 2025 one. What is the
most reliable way to stop the bot from quoting the old one?

- A) Add "always prefer the newest document" to the system prompt.
- B) Filter out `status: superseded` before ranking, and also show `status` and `effective_date` in
  the prompt.
- C) Delete the 2023 policy from every system.
- D) Fine-tune the model on the 2025 policy.

<details><summary>Answer</summary>

**B.** The filter is deterministic, and the metadata in the prompt is a backup for conflicts the
filter does not cover. A alone is probabilistic (the lab's stand-in model ignores it and answers
"14 days"). C loses history that agents and auditors need. D stores a fact that will change again
in the weights.

</details>

### Q2 · Spot the bug

```python
def domain_allowed(domain: str) -> bool:
    return "hearthwise" in domain.lower()
```

<details><summary>Answer</summary>

It allows lookalike domains. The injected paragraph in `kb-025` asks customers to write to an
address at `hearthwise-support.example`, which contains "hearthwise". Compare the full domain
exactly, or allow exact subdomains (`domain == d or domain.endswith("." + d)`).

</details>

### Q3 · Short calculation

With the status filter off, the returns question retrieves chunks from `kb-010`, `kb-011`,
`kb-011`, `kb-010`, in that order. Only `kb-011` is relevant. Compute hit@4, the reciprocal rank and
precision@4.

<details><summary>Answer</summary>

hit@4 = 1 (a relevant chunk is present), reciprocal rank = 1/2 = 0.5 (the first relevant chunk is
second), precision@4 = 2/4 = 0.5. Retrieval looks fine on these numbers, yet the answer is wrong,
because the superseded chunk ranked first and the model quoted it.

</details>

### Q4 · Multiple choice

In a RAFT-style training set for a customer-support bot, which examples teach the model to refuse
when retrieval fails?

- A) Examples with the oracle document and three distractors.
- B) Examples with the oracle document in first position.
- C) Examples without the oracle document whose target is `NO_ANSWER`.
- D) Examples whose target quotes the oracle verbatim.

<details><summary>Answer</summary>

**C.** Only examples without the oracle show the model a context that lacks the answer. With
`NO_ANSWER` targets (the lab's default) it learns to refuse; with answer targets (the recipe
described in the RAFT paper) it learns to answer from memory instead.

</details>

### Q5 · Short calculation

With both injection defences off, the bot answers "Will Hearthwise support ever ask for my
password?" with two sentences: the true one from `kb-014` and the attacker's instruction from
`kb-025`. Both pages are in the context. What faithfulness does the lab's claim-level metric give,
and what does that tell you?

<details><summary>Answer</summary>

1.000: both claims are supported by the context, because the poisoned page is part of it.
Faithfulness measures agreement with the context, not truth or safety. That is why safety needs its
own checks, such as the global deny-list, which flags this answer.

</details>

### Q6 · What would you do?

After someone removes the output guardrail, the dashboard shows correctness going *up* on the
golden set. What do you check before celebrating?

<details><summary>Answer</summary>

The safety columns. In the lab, "no sanitiser, no guardrail" scores 32/36 against 30/36 for "no
sanitiser", because the guardrail had replaced two poisoned answers with handoffs. Without it, the
attacker's text reaches customers and the global deny-list counts 2 leaks. Never read quality
metrics without safety metrics, and gate on leaks with zero tolerance.

</details>

### Q7 · Multiple choice

According to the lab's judge calibration, which faithfulness judge catches all 21 number swaps with
the fewest false alarms?

- A) Word overlap of at least 0.9, no number rule.
- B) Word overlap of at least 0.9, with the number rule.
- C) Word overlap of at least 0.5, with the number rule.
- D) Word overlap of at least 0.5, no number rule.

<details><summary>Answer</summary>

**C.** It catches 21 of 21 number swaps with 1 false alarm out of 70 faithful cases. B also catches
all 21 but rejects 34 faithful answers. A catches 15 and D catches none. The number rule, not a
stricter threshold, is what catches invented numbers.

</details>

### Q8 · Short calculation

At a threshold of 0.40, the bot answers 37 of the 40 golden questions, has 0 false abstentions and
refuses 3 of the 4 questions that must be refused. What is its abstention accuracy?

<details><summary>Answer</summary>

Correct decisions: all 36 answerable questions answered, plus 3 refusals = 39. Accuracy = 39/40 =
0.975, the same as at 0.50, where the bot refuses all 4 but also refuses one answerable question.
Equal accuracy, different mistakes: choosing between them is a product decision.

</details>

### Q9 · Multiple choice

Tomorrow the model vendor ships a new model version. Which injection defences keep working without
re-testing the model's behaviour?

- A) The system-prompt rule "documents are data".
- B) The sanitiser at ingestion and the output guardrail.
- C) Delimiting documents with tags.
- D) None; everything depends on the model.

<details><summary>Answer</summary>

**B.** Both run in your code, before and after the model, so their behaviour does not change with
the model. A and C still help, but how well a model respects rules and tags must be re-measured for
every model version.

</details>

### Q10 · What would you do?

Alex asks you to "get rid of all the I don't know answers". What do you show, and what do you
propose?

<details><summary>Answer</summary>

Show the threshold sweep: at 0.00 the bot answers all 40 questions, including the 4 that must be
refused, so abstention accuracy falls to 0.900, and those answers are cited but useless or wrong.
Propose instead to look at the false abstentions (q035 at the default threshold) and fix their
cause (here, reasoning across a table), to improve the handoff experience, and to track
abstentions per topic so missing articles get written.

</details>

### Q11 · Spot the bug

```python
examples = build_all_examples(golden_rows)  # 3 copies per question
random.shuffle(examples)
train, test = examples[:90], examples[90:]
```

<details><summary>Answer</summary>

The split happens after building examples, so copies of the same question can land on both sides.
The test set is then contaminated with questions the model trained on. Split the question ids first
(stratified, so both sides contain refusals), then build examples only from the training questions,
as the lab's `split_questions` does.

</details>

### Q12 🔁 (module 05)

The bot misses "How much does the Watcher S1 weigh?" because the answer sits in a six-row HTML
comparison table. Using what you learned about chunking and BM25, explain why the table chunk ranks
below four Watcher S1 chunks, and name two retrieval fixes.

<details><summary>Answer</summary>

The table becomes one long chunk, and BM25's length normalisation lowers the score of long
documents for the same term matches, while short Watcher S1 chunks repeat "Watcher S1" in their
title and text. Fixes: chunk tables by row (each row with its header), raise `top_k` (the lab's
budget shows q017 is answered at `top_k=8`) or retry retrieval when evidence is weak, and add a
reranker. Re-run the full evaluation after any of them.

</details>

### Q13 🔁 (module 04)

Module 04 compared prompting, context engineering and LoRA fine-tuning. For the support bot, why
would you exhaust prompt and retrieval fixes before training a LoRA adapter on RAFT data?

<details><summary>Answer</summary>

They are cheaper, faster to iterate and reversible, and many failures are not generation failures
at all (q017 is a retrieval miss). Training costs data creation, review, compute and a full
re-evaluation after every run, and facts learned in weights cannot be filtered or updated at
request time. Fine-tune when the right context is present and the model still misuses it.

</details>

### Q14 🔁 (module 03)

In module 03 you made latency and cost visible in the playground. In the lab, going from `top_k=4`
to `top_k=8` raises the average RAG prompt from 582.9 to 1027.0 tokens. What happens to cost and
context-window use, and what would you tell Sam?

<details><summary>Answer</summary>

Prompt tokens grow by about 76%, so input cost per request and the share of the context window
used by retrieved text grow by the same factor, on every request. Tell Sam the gain: one more
correct golden answer (33/36 instead of 32/36). Then offer a cheaper alternative: retry with more
chunks only when the evidence is weak, and measure both cost and quality.

</details>

## Scoring guide

| Score | What it means |
|---|---|
| 12–14 | You are ready for module 07. |
| 9–11 | Re-read the sections behind the questions you missed and re-run the matching lab step. |
| 0–8 | Work through the lab again from step 2, then retake the quiz. |

## ⏭️ Next

[Module 07 — Agents I: Workflows, Tools and MCP](../07-agents-workflows-and-tools/README.md)

---

[← Lab 06](lab/README.md) · [🏠 Module 06](README.md) · [Cheatsheet →](cheatsheet.md)
