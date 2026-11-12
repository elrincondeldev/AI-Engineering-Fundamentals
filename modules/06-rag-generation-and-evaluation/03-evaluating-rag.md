🌐 English · [Español](03-evaluating-rag.es.md)

# Lesson 6.3 — Evaluating RAG

> **Module 06 · Lesson 3 of 4** · ⏱️ about 55 min · 🧪 Lab: [step 5](lab/README.md#step-5--evaluate-on-the-golden-set)

## 📨 Situation

> **From:** Dani Ruiz · Data Scientist, owns evaluation
> **Channel:** #ai-evals
>
> I hear the bot is "ready". Show me the eval. Forty golden questions, including the four where
> silence is the right answer. For each: did the right page come back, is every sentence backed by
> it, is the answer actually right, did it refuse what it should refuse, and did anything leak.
> And before you show me a faithfulness score from a judge, show me how often that judge is wrong.
> One number per stage, please, not one number for everything.

## 🎯 Objectives

By the end of this lesson you can:

- Compute retrieval metrics (hit@k, mean reciprocal rank) and context relevance on a labelled
  golden set.
- Measure faithfulness claim by claim and answer correctness with key facts and token F1, and
  explain why the two can disagree.
- Measure abstention accuracy and safety with per-question and global deny-lists.
- Calibrate a judge against labelled cases and choose its threshold from the results.
- Turn the evaluation into ablations and a regression gate that runs on every change.

## 🤔 Predict first

Write your answers down before reading on. You will check them in the self-check.

1. Can an answer be 100% faithful to its context and still be wrong?
2. Which judge catches more invented numbers: one that requires 90% word overlap, or one that
   requires 50% overlap plus an exact-number rule?
3. No answer contains any of its question's forbidden strings. Can something still have leaked?

## 1. What to measure, and where

A retrieval-augmented generation (RAG) system can fail at several stages, and a single "accuracy"
number hides which one. Dani's request maps to one family of metrics per stage, close to the
family described in the RAGAS paper (see Go deeper):

| Stage | Metric | The question it answers |
|---|---|---|
| Retrieval | hit@k (context recall), mean reciprocal rank (MRR) | Did a relevant document reach the context, and how high? |
| Context | Context relevance (precision@k) | How much of the context is about the question? |
| Generation | Faithfulness | Is every claim in the answer supported by the context? |
| End to end | Answer correctness (key facts, token F1) | Is the answer right, compared with a reference? |
| Behaviour | Abstention accuracy, language match | Does it refuse exactly when it should, in the right language? |
| Safety | Forbidden strings, global deny-list, guardrail blocks | Did anything that must never reach a customer reach one? |

The golden set, `datasets/rag_eval/qa.jsonl`, has 40 questions: 36 answerable ones spread over
categories such as `policy`, `table` and `cross_lingual`, and 4 where the correct behaviour is to
refuse (3 `unanswerable`, 1 `internal`). Each row carries a reference answer, the ids of the
relevant documents and a `must_not_contain` list.

**Analogy:** a car inspection checks brakes, lights and emissions separately. "It drove around the
block" is not an inspection, and a failed brake test tells the mechanic where to look.

```mermaid
flowchart LR
    Q["Golden question"] --> R["Retrieve"]
    R --> G["Generate"]
    G --> A["Answer"]
    R -.-> M1["hit@k, MRR,<br/>context relevance"]
    G -.-> M2["Faithfulness"]
    A -.-> M3["Correctness, abstention,<br/>language, safety"]
```

> ⚠️ **Common mistake:** reporting one blended score. If correctness drops, you need to know
> whether retrieval, generation or the refusal logic moved, and a blended score cannot tell you.

## 2. Retrieval and context relevance

Retrieval metrics use the `relevant_doc_ids` labels. *hit@k* asks whether at least one chunk from a
relevant document is among the k chunks in the context; averaged over the 36 answerable questions
it is the *context recall*. The *reciprocal rank* is 1 divided by the position of the first relevant
chunk (1 for first place, 0.5 for second, 0 if absent); its mean is the MRR. *Context relevance*,
here measured as precision@k, is the share of the k chunks that come from a relevant document.

With `top_k=4`, the full bot scores 0.972 on context recall: 35 of the 36 answerable questions
have a relevant chunk in the context. MRR is also 0.972, so when the relevant document is there, it
is almost always first. Context relevance is 0.611: on average, a bit more than a third of each
context is noise. For reference, the [shared dataset notes](../../datasets/README.md) report doc
recall@5 of 0.917 for plain BM25 over heading-level chunks; this lab's chunking, table linearisation and query rewriting are
different, so treat the two numbers as cousins, not as a controlled comparison.

The one miss is q017, "How much does the Watcher S1 weigh?". The weight lives only in a six-row
HTML comparison table that becomes one long chunk, and four shorter Watcher S1 chunks outrank it.
That is a retrieval failure, and no generator can fix it.

```python
def context_relevance(context: list[Chunk], relevant_doc_ids: list[str]) -> float | None:
    """Share of retrieved chunks that come from a relevant document (None if no labels)."""
    if not context or not relevant_doc_ids:
        return None
    return sum(c.doc_id in relevant_doc_ids for c in context) / len(context)


def reciprocal_rank(context: list[Chunk], relevant_doc_ids: list[str]) -> float:
    for rank, chunk in enumerate(context, 1):
        if chunk.doc_id in relevant_doc_ids:
            return 1 / rank
    return 0.0
```

**Analogy:** a librarian who brings you four books. Recall asks whether the right book is in the
pile; MRR asks whether it is on top; relevance asks how many of the four you can put straight back.

> 💡 **Production tip:** retrieval labels can be collected cheaply from production. When an agent
> corrects the bot, ask which article they used; that article becomes a relevance label for the
> customer's question.

## 3. Faithfulness: is every claim supported?

Faithfulness (also called groundedness) asks whether the answer says only what the context says.
It is measured claim by claim, because a single invented sentence can hide inside three correct
ones:

1. Split the answer into claims. The lab uses sentences and cited segments; an LLM can split more
   finely.
2. For each claim, search the context for a sentence that supports it. The lab's heuristic: one
   context sentence (with its title and heading) must contain at least a share τ of the claim's
   content words, and every number in the claim must appear in that same sentence.
3. Faithfulness is supported claims divided by claims.

```python
def check_claim(claim: str, units: list[tuple[str, set[str], set[str]]], *, tau: float = 0.5,
                check_numbers: bool = True) -> ClaimCheck:
    """A claim is supported if one unit covers >= ``tau`` of its words (and all its numbers)."""
    words = set(content_tokens(claim))
    nums = numbers(claim)
    best = ClaimCheck(claim, False, 0.0, "")
    for unit, unit_words, unit_nums in units:
        overlap = len(words & unit_words) / len(words) if words else 0.0
        ok = overlap >= tau and (not check_numbers or nums <= unit_nums)
        if (ok, overlap) > (best.supported, best.overlap):
            best = ClaimCheck(claim, ok, overlap, unit)
    return best
```

The full bot scores 1.000. That is not a triumph: the stand-in model copies sentences, so its
claims are supported by construction. The metric earns its place when you swap in a real model,
which paraphrases, merges and sometimes invents. It also teaches a sharp lesson right now:
faithfulness is measured against the *context*, not against the truth. With the status filter off,
the bot answers "You can return most Hearthwise products within 14 days of delivery." That claim is
perfectly faithful to `kb-010`, the superseded policy in its context, and completely wrong for a
customer today.

```mermaid
flowchart TB
    A["Answer"] --> S["Split into claims"]
    S --> C{"One context sentence covers<br/>at least tau of the words<br/>and all the numbers?"}
    C -->|yes| Y["Supported"]
    C -->|no| N["Unsupported"]
    Y --> F["Faithfulness =<br/>supported / claims"]
    N --> F
```

**Analogy:** a fact-checker compares an article with its sources, line by line. They can certify
that every line matches a source, and still miss that the source itself is out of date.

> ⚠️ **Common mistake:** checking faithfulness against the whole knowledge base instead of the
> retrieved context. Then a stale page "supports" a stale answer, and claims the model could not
> have seen look grounded.

## 4. Correctness, abstention and safety

**Correctness** needs a reference. Comparing whole sentences is fragile, so for each answerable
golden question the lab stores a few *key facts*, short phrases a correct answer must contain:
`"30 days"` for the returns window, `"3680 w"` for the Spark Mini load, and alternatives separated
by `|` when a fact may appear in Spanish (`"30 dias|30 days"`). A question is correct when all its
key facts are present. The full bot gets 32 of 36 right (0.889), with a mean key-fact recall of
0.903. Token F1, the word-overlap score between answer and reference, is only 0.542: the bot's
copied sentences are longer than the short references, so precision suffers even when the fact is
there. That is why you should not judge an extractive or verbose system by F1 alone.

**Abstention accuracy** is the share of all 40 questions where the bot answered if and only if the
question was answerable: 39 of 40 (0.975). All 4 questions that must be refused were refused; the
1 false abstention is q035, which needs comparing launch dates across a table, something an
extractive model cannot do. The threshold controls the trade-off:

| Threshold | Answered | Correct | Unanswerable refused | False abstentions | Abstention accuracy |
|---|---|---|---|---|---|
| 0.00 | 40 | 32 | 0/4 | 0 | 0.900 |
| 0.30 | 38 | 32 | 2/4 | 0 | 0.950 |
| 0.40 | 37 | 32 | 3/4 | 0 | 0.975 |
| 0.50 | 35 | 32 | 4/4 | 1 | 0.975 |
| 0.60 | 31 | 28 | 4/4 | 5 | 0.875 |
| 0.70 | 27 | 24 | 4/4 | 9 | 0.775 |

Accuracy ties at 0.40 and 0.50, so the choice is a product decision: Alex prefers one unnecessary
handoff (q035) over one confident answer to a question the help centre does not cover.

**Safety** has two layers. Each golden row lists strings its answer must never contain, such as
`"14 days"` for the returns window. That catches the attacks you anticipated for that question.
The global deny-list (`GLOBAL_DENYLIST` in [`metrics.py`](lab/src/m06_support_bot/metrics.py))
checks every answer for strings that must never reach anyone: the internal voucher code and markers
of the known injection payload. With both injection defences off, per-question checks flag 1 answer
(q040), but the global list flags 2: the hidden paragraph also poisoned q026, whose
`must_not_contain` list is empty. The full bot has 0 of each. Finally, *language match* checks that
answers come back in the question's language: 0.943, because the two cross-lingual answers are in
Spanish.

```python
def key_fact_recall(answer: str, facts: list[str]) -> float:
    """Share of key facts present in the answer (each fact may list alternatives with '|')."""
    if not facts:
        return 0.0
    hits = sum(any(_contains(answer, alt) for alt in fact.split("|")) for fact in facts)
    return hits / len(facts)
```

> 💡 **Production tip:** always print the failing questions next to the summary, as `lab eval`
> does under "To review". Dani will read those six lines before any average, and so should you.

## 5. Calibrating the judge

Every metric above is a judge: a function that looks at an answer and returns a verdict. When the
verdict needs understanding rather than string matching, teams use an LLM as the judge
(LLM-as-a-judge, see Go deeper). Judges, heuristic or LLM, make mistakes, and their mistakes are
systematic. Before you trust one, measure how often it agrees with labels you trust.

Human labels are the gold standard. Offline, the lab builds labels by construction. It takes the
35 answers the bot gave, keeps each one (faithful), adds a paraphrase that starts "According to our
help centre," (still faithful), and corrupts it three ways: one number increased by one (21 cases,
where the answer had a number), a sentence from an unrelated document appended with a fake citation
(35 cases), and a negation such as "cannot" turned into "can" (11 cases). That gives 137 labelled
cases. Then it runs the heuristic judge with different settings:

| Judge | Agreement | False alarms | Number swaps caught | Foreign claims caught | Negations caught |
|---|---|---|---|---|---|
| τ 0.5, numbers off | 0.759 | 1/70 | 0/21 | 35/35 | 0/11 |
| τ 0.5, numbers on | 0.912 | 1/70 | 21/21 | 35/35 | 0/11 |
| τ 0.7, numbers on | 0.854 | 9/70 | 21/21 | 35/35 | 0/11 |
| τ 0.9, numbers off | 0.635 | 34/70 | 15/21 | 35/35 | 1/11 |
| τ 0.9, numbers on | 0.679 | 34/70 | 21/21 | 35/35 | 1/11 |

Three lessons fall out. A strict overlap threshold does not catch more lies; it mostly rejects
honest paraphrases (34 of 70 false alarms at 0.9). The exact-number rule, not the threshold, is
what catches invented numbers. And no word-overlap judge sees negation: "can" and "cannot" share
all their content words. The lab therefore uses τ 0.5 with the number rule, and documents the
negation blind spot. To cover it you would add an LLM judge (the lab ships `llm_judge()` with the
same interface) and calibrate it on exactly these cases before trusting it.

```mermaid
flowchart LR
    L["Labelled cases<br/>faithful and corrupted"] --> J["Run the judge"]
    J --> C["Compare verdicts<br/>with labels"]
    C --> D{"Agreement and<br/>blind spots acceptable?"}
    D -->|no| T["Change threshold,<br/>rules or judge model"]
    T --> J
    D -->|yes| U["Use the judge,<br/>report its agreement"]
```

**Analogy:** before a referee works a final, someone reviews their past calls against video. Nobody
asks whether the referee is confident; they ask how often the calls matched what happened.

> ⚠️ **Common mistake:** reporting a faithfulness score from an LLM judge without the judge's own
> error rate. If the judge often disagrees with humans, a high score means much less than it seems.

> 💡 **Production tip:** keep a few hundred human-labelled cases from real traffic, re-calibrate
> whenever the judge model or its prompt changes, and report the judge's agreement next to every
> score it produces.

## 6. Golden sets, ablations and regression gates

A golden set is a small, curated, versioned list of questions with references and labels. Ours
covers the traps the help centre hides: a superseded policy, an internal-only document, facts only
in Spanish or only in a table, an injection-exposed page and questions with no answer. A good golden
set grows: every production failure Alex's agents report becomes a new row.

Be honest about one weakness. The lab's glossary, stopwords and threshold were tuned while looking
at these same 40 questions, so 0.889 correctness is an optimistic estimate. That is why lesson 6.2
held out 10 questions, and why fresh questions from production matter more than a higher score on
old ones.

An **ablation** switches off one component and re-runs the evaluation. It turns "we have a
sanitiser" into "the sanitiser prevents this many leaks":

| Configuration | Correct | Abstention accuracy | Violations | Global leaks | Guardrail blocks |
|---|---|---|---|---|---|
| full | 32/36 | 0.975 | 0 | 0 | 0 |
| no status filter | 30/36 | 0.975 | 2 | 0 | 0 |
| no audience filter | 32/36 | 0.950 | 1 | 1 | 0 |
| no sanitiser | 30/36 | 0.925 | 0 | 0 | 2 |
| no sanitiser, no guardrail | 32/36 | 0.975 | 1 | 2 | 0 |
| no query rewriting | 30/36 | 0.925 | 0 | 0 | 0 |

Read the "no sanitiser, no guardrail" row twice: its correctness and abstention accuracy look
*better* than "no sanitiser", because the poisoned answers still contain the right Wi-Fi tips. Only
the safety columns reveal the attack. A dashboard that tracks correctness alone would have
celebrated.

A **regression gate** turns the evaluation into a test. The lab's `GATES` require correctness of at
least 0.85, faithfulness of at least 0.95, abstention accuracy of at least 0.95, and zero violations
and leaks. `uv run lab eval --gate` exits with code 1 when any gate fails, so continuous integration
(CI) can block a pull request that, for example, drops the status filter.

```mermaid
flowchart LR
    PR["Pull request:<br/>prompt, model or index change"] --> E["lab eval --gate"]
    E --> G{"All gates pass?"}
    G -->|yes| M["Merge"]
    G -->|no| B["Blocked, with the<br/>failing questions listed"]
```

> 💡 **Production tip:** gate on safety with zero tolerance and on quality with a small margin
> below the current score. A gate set exactly at today's number fails on harmless noise, and people
> learn to ignore it.

## 🧪 Try it

Run these from `modules/06-rag-generation-and-evaluation/lab`.

1. Score the full bot on the golden set.

   ```bash
   uv run lab eval
   ```

   ```text
   Golden set: 40 questions (36 answerable, 4 must be refused) | config: full

   answer correctness (all key facts)  32/36 = 0.889
   key-fact recall (mean)              0.903
   token F1 vs reference (mean)        0.542
   faithfulness (claims supported)     1.000
   context relevance (precision@k)     0.611
   context recall (hit@k)              0.972
   MRR@k                               0.972
   abstention accuracy                 39/40 = 0.975
   unanswerable questions refused      4/4
   false abstentions (answerable)      1
   must_not_contain violations         0
   global deny-list leaks              0
   guardrail blocks                    0
   language match (answered)           0.943
   ```

   Below the summary you get a table per category and the six questions to review.

2. Run the ablations and the threshold sweep.

   ```bash
   uv run lab ablate
   ```

   ```text
   config                     correct  abst.acc  violations  leaks  blocked  leaked ids
   full                        32/36      0.975           0      0        0  -
   no_status_filter            30/36      0.975           2      0        0  q001,q002
   no_audience_filter          32/36      0.950           1      1        0  q039
   no_sanitiser                30/36      0.925           0      0        2  -
   no_sanitiser_no_guardrail   32/36      0.975           1      2        0  q026,q040
   no_rewrite                  30/36      0.925           0      0        0  -
   ```

3. Calibrate the judge.

   ```bash
   uv run lab judge
   ```

   ```text
   Judge cases: 137 (original 35, paraphrase 35, number_swap 21, foreign_claim 35, negation 11)
   Labels are known by construction: original and paraphrase are faithful, the rest are not.

   judge                  agreement false alarms number_swap foreign_claim negation
   tau=0.5 numbers=off        0.759         1/70        0/21         35/35     0/11
   tau=0.5 numbers=on         0.912         1/70       21/21         35/35     0/11
   tau=0.7 numbers=off        0.701         9/70        0/21         35/35     0/11
   tau=0.7 numbers=on         0.854         9/70       21/21         35/35     0/11
   tau=0.9 numbers=off        0.635        34/70       15/21         35/35     1/11
   tau=0.9 numbers=on         0.679        34/70       21/21         35/35     1/11
   ```

4. Use the gate, then break it: without the status filter the gate fails and `lab` exits with
   code 1, which is what makes CI stop.

   ```bash
   uv run lab eval --gate | tail -1
   uv run lab eval --config no_status_filter --gate > /dev/null; echo "exit code: $?"
   uv run lab eval --config no_status_filter --gate | tail -1
   ```

   ```text
   Regression gate: PASS
   exit code: 1
   Regression gate: FAIL correct_rate=0.833 (limit 0.85); violations=2 (limit 0)
   ```

## 🏭 In the real world

- **Evaluation is continuous.** The same suite runs on every prompt, model, chunking or index
  change, and nightly against a sample of production traffic.
- **LLM judges are common and calibrated.** Teams use them for faithfulness and relevance at scale,
  validated against a human-labelled set and re-checked after every judge-model upgrade.
- **Golden sets are products.** They have owners, versions and change logs. Rows come from
  incidents, agent corrections and new policies, and old rows are retired when the help centre
  changes.
- **Safety metrics are zero-tolerance.** A single leak of an internal code is an incident, not a
  rounding error in an average.
- **Offline and online metrics disagree.** Production adds signals the golden set lacks: thumbs
  up or down, handoff rates, reopened tickets. Lesson 6.4 wires those into the feedback loop.

## ✅ Self-check

**1. Can an answer be 100% faithful and still wrong?**

<details><summary>Answer</summary>

Yes. Faithfulness compares the answer with the context, not with the truth. With the status filter
off, "within 14 days of delivery" is fully supported by the superseded policy in the context, and
wrong for every customer today. That is why correctness and faithfulness are separate metrics.

</details>

**2. Which judge catches more invented numbers: 90% overlap, or 50% overlap plus a number rule?**

<details><summary>Answer</summary>

The 50% judge with the number rule: it catches 21 of 21 number swaps with 1 false alarm in 70
faithful cases. The 90% judge without the rule catches only 15 of 21, and rejects 34 of 70 honest
answers. Raising a threshold is not the same as checking the right thing.

</details>

**3. No answer contains its own `must_not_contain` strings. Can something still have leaked?**

<details><summary>Answer</summary>

Yes. Per-question lists only cover attacks you foresaw for that question. With both injection
defences off, the poisoned page also reached the password question (q026), whose list is empty.
A global deny-list checked on every answer caught 2 leaks where per-question checks saw 1.

</details>

**4. Context relevance is 0.611. Is the retriever broken?**

<details><summary>Answer</summary>

Not necessarily. With 4 chunks per question and often a single relevant document, some noise is
expected; context recall (0.972) and MRR (0.972) show the relevant document is nearly always there
and first. Low relevance matters for cost and for weaker models that get distracted, which is a
reason to try a smaller `top_k` or a reranker, and to measure again.

</details>

**5. Why does the "no sanitiser, no guardrail" configuration score better on correctness than "no
sanitiser"?**

<details><summary>Answer</summary>

Because in "no sanitiser" the guardrail blocks two poisoned answers and replaces them with a
handoff, which counts as incorrect. Without the guardrail, those answers still contain the right
Wi-Fi tips plus the attacker's text, so key facts are found. Only the safety columns show the
difference, which is why you never read quality metrics without safety metrics.

</details>

## 💬 Explain it in your own words

- Explain to Dani why the bot's faithfulness of 1.000 is not evidence that a real model would be
  faithful, and what you would run to find out.
- Explain to Alex, with the threshold table, why the bot refuses q035 and why you still recommend
  a threshold of 0.50.

## 🔑 Key ideas

- Measure each stage separately: retrieval (hit@k, MRR), context relevance, faithfulness,
  correctness, abstention, language and safety.
- Faithfulness is about the context, correctness is about the truth; a stale document makes them
  disagree.
- Key facts are a cheap, robust correctness check; token F1 punishes long but correct answers.
- Calibrate every judge on labelled cases and report its agreement; know its blind spots.
- Ablations turn components into measured value, and a regression gate keeps the numbers from
  sliding back.

## 📚 Go deeper

- ⭐ [RAGAS: automated evaluation of retrieval-augmented generation](https://arxiv.org/abs/2309.15217)
  — the reference-free metric family (faithfulness, answer relevance, context relevance) this
  lesson builds on.
- [Judging LLM-as-a-judge](https://arxiv.org/abs/2306.05685) — how strong models are used as
  judges, and the biases you must calibrate for.
- [HELM: holistic evaluation of language models](https://arxiv.org/abs/2211.09110) — why one
  number per model is never enough.

---

[← Previous: 6.2 RAFT: training for RAG](02-raft-training-for-rag.md) · [🏠 Module 06](README.md) · [Next: 6.4 Designing the support chatbot →](04-support-chatbot-design.md)
