🌐 English · [Español](quiz.es.md)

# Module 03 quiz — Build an LLM Playground

Thirteen questions. Answer each one on paper before you open the answer. Questions marked 🔁 review
earlier modules.

### Q1 — Which component keeps the promise?

Noor requires that no customer personal data reaches an external API (application programming
interface). Where must the playground enforce this?

- A. In the web client, by hiding the external models until redaction is switched on.
- B. In the orchestrator, by redacting before prompt assembly, storage and logging, and refusing external providers when redaction is off.
- C. In the system prompt: "Never repeat personal data."
- D. In the trace log, by deleting lines that contain email addresses.

<details><summary>Answer</summary>

**B.** Every request passes the orchestrator, so a check there cannot be bypassed. A client-side
check (A) is skipped by any script that calls the API directly. A system prompt (C) asks the model to
behave but does not stop the data from being *sent*. Cleaning the logs (D) happens after the data has
already left.

</details>

### Q2 — Latency by hand

The `fake-small` profile has 120 ms of overhead, 0.25 ms of prefill per prompt token and 20 ms per
generated token after the first. A request has a 120-token prompt and a 41-token answer. Compute the
time to first token (TTFT), the total latency, the decode speed and the end-to-end throughput.

<details><summary>Answer</summary>

- TTFT = 120 + 0.25 × 120 = **150 ms**.
- Total = 150 + 40 × 20 = **950 ms**.
- Decode speed = 40 tokens ÷ 0.8 s = **50 tokens/s**.
- End-to-end = 41 tokens ÷ 0.95 s ≈ **43.2 tokens/s**.

The end-to-end number is lower because it includes the wait before the first token. That is why it
should not be reported as "model speed".

</details>

### Q3 — The monthly bill

Lesson 3.1's workload sends 31,500 requests a month with 180 completion tokens each. On `fake-large`
(example prices: €1.00 per million input tokens, €4.00 per million output tokens) the bill was €44.73
with 700 prompt tokens per request. Conversations get longer and the average prompt grows to 1,400
tokens. What is the new monthly cost?

<details><summary>Answer</summary>

Per request: (1,400 × 1.00 + 180 × 4.00) ÷ 1,000,000 = €0.00212. Per month: 0.00212 × 31,500 =
**€66.78**. Doubling the prompt did not double the bill, because the output part (€0.00072 per
request) did not change. On long conversations, though, input dominates and keeps growing.

</details>

### Q4 — Spot the bug: a retry loop

```python
def call_with_retries(provider, messages):
    while True:
        try:
            return provider.complete(messages)
        except Exception:
            time.sleep(1)
```

Name at least three problems.

<details><summary>Answer</summary>

1. It never gives up: no maximum number of retries and no deadline.
2. It retries everything, including errors that will never succeed (`bad_request`, `context_length`)
   and errors that need a different provider (`auth`).
3. A fixed one-second wait has no exponential backoff and no jitter, so many clients retry in
   synchronised waves.
4. It ignores `Retry-After`, never falls back, and records nothing about why the call failed.

Compare it with `ModelGateway.complete` in the lab's `resilience.py`.

</details>

### Q5 — Retry or not?

Which failure should the gateway **not** retry?

- A. HTTP 429 with `Retry-After: 2`.
- B. HTTP 503.
- C. A timeout after 10 seconds.
- D. HTTP 400, "maximum context length exceeded".

<details><summary>Answer</summary>

**D.** The same prompt fails the same way every time; the fix is to shorten it (lesson 3.3). A, B
and C are transient and worth retrying with backoff, A after the two seconds the server asked for.

</details>

### Q6 — Retries in numbers

A provider fails 20% of calls, independently of each other. The gateway allows up to 2 retries. What
share of requests succeed? Why might a real system measure less than that?

<details><summary>Answer</summary>

A request fails only if all three attempts fail: 1 − 0.2³ = **99.2%** succeed. Real systems measure
less when failures are *not* independent (an outage fails every attempt, as the lab's outage rows
show), and when slow failures such as timeouts use up the deadline before the last retry.

</details>

### Q7 — Input budget

A model has a context window of 4,096 tokens. You set `max_tokens` to 512, and your system prompt is
300 tokens. How many tokens are left for the history and the new message?

<details><summary>Answer</summary>

Input budget = 4,096 − 512 = 3,584 tokens. Minus the 300-token system prompt: **3,284 tokens** for
the history and the new message. Forgetting to subtract `max_tokens` is the classic mistake: the
prompt fits, then the request fails or the answer is cut.

</details>

### Q8 — The forgotten order

Long support chats in the playground forget the order number the customer gave in the first turn. The
`pin_system` policy is active. What do you change?

- A. Switch to a sliding window of the last six messages.
- B. Switch to the `summarise` policy, and keep order numbers in a structured field of the session.
- C. Raise `max_tokens`.
- D. Nothing: tell agents to repeat the order number.

<details><summary>Answer</summary>

**B.** In the lab's replay, `summarise` still knew `#HW-640183` at turn 20 for 82 more tokens than
`pin_system`. A sliding window (A) forgets even sooner and drops the system prompt too. Raising
`max_tokens` (C) *shrinks* the input budget. D moves the cost to customers.

</details>

### Q9 — Spot the bug: truncation

```python
def build_prompt(system, history, user):
    messages = [system, *history, user]
    return messages[-6:]
```

<details><summary>Answer</summary>

The system prompt is the oldest message, so it is the first to fall out: from the fourth turn on the
model runs without its rules (17 of 20 turns in the lab's replay). It also counts messages instead of
tokens, so six long messages can still overflow the window, and it can cut a user/assistant pair in
half. Fix: pin the system prompt, fill the rest with whole pairs by token budget.

</details>

### Q10 — Reading the redactor table

The `regex` level scores: emails 1224/1224, phones 82/82, names 0/82, recall 0.941, leaking tickets
82. Which statement is correct?

- A. A recall of 0.941 means that 94% of tickets are safe to send.
- B. Every ticket with a signature still leaks a name; the overall recall hides a complete miss of one entity type.
- C. The `regex` level has low precision.
- D. The 82 leaking tickets are caused by dates.

<details><summary>Answer</summary>

**B.** Recall counts items, not tickets, and 1,224 easy emails swamp 82 names. The `regex` level's
precision is 1.000 (C is wrong); the dates were the `naive` level's false positives (D is wrong).
Always report per entity type.

</details>

### Q11 — "Just for debugging"

A teammate adds the raw prompt to every trace line "temporarily", to debug a strange answer. What do
you do?

<details><summary>Answer</summary>

Remove it. Raw prompts turn the traces into an unplanned copy of customer data, with no retention
period or access control, and "temporary" fields tend to stay. Debug with what the trace already has
(`prompt_hash`, parameters, attempts, flags) and reproduce the case with the *redacted* prompt. If raw
text is truly needed, agree an exception with Noor: a separate store, restricted access and a short
retention period.

</details>

### Q12 🔁 (module 02) — Temperature

In compare mode, `fake-small` gave 1 distinct answer over 10 seeds at temperature 0 and 3 at
temperature 1.5, and 2 of those 10 repeated the outdated 14-day returns policy. Using what you learned
about decoding, explain what temperature does and why the wrong answer appeared.

<details><summary>Answer</summary>

Temperature divides the model's scores (logits) before the softmax. Near 0 the distribution collapses
onto the most likely option, which is greedy decoding. Above 1 it flattens, so unlikely options get
picked more often. The outdated answer was always in the distribution with a low probability;
temperature 1.5 gave it a real chance. Temperature adds variety, not knowledge, so factual questions
stay at 0 and output checks act as a safety net.

</details>

### Q13 🔁 (module 01) — Counting tokens

The lab counts tokens as words plus punctuation marks. Why will a provider's count for the same prompt
be different, and which count should you use for budgets and billing?

<details><summary>Answer</summary>

Real models use subword tokenizers (such as byte-pair encoding) learned from training data: rare
words, numbers, Spanish text and code often split into several tokens, and chat templates add special
tokens for the roles. Use the model's own tokenizer, or the usage the provider returns with each
response, for budgets and cost. The lab's approximation is only for teaching the shape of the growth.

</details>

## Scoring guide

| Correct answers | What to do next |
|---|---|
| 12–13 | You are ready for module 04. |
| 9–11 | Re-read the sections behind the questions you missed and rerun the matching lab step. |
| 0–8 | Redo lab steps 1 to 5, then take the quiz again in a day or two. |

## ⏭️ Next

[Module 04 — Adapting LLMs: prompting, context and fine-tuning](../04-prompting-context-and-fine-tuning/README.md)
