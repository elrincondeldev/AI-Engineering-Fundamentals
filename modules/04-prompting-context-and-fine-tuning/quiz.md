🌐 English · [Español](quiz.es.md)

# Module 04 quiz — Adapting LLMs

Answer each question on paper first, then open the answer. Questions marked 🔁 review earlier modules.
Every number comes from the module 04 lab.

### Q1

A router will run on next month's tickets. Which evaluation design gives the most honest estimate of
its accuracy?

- A. A random 80/20 split of all 1,224 tickets.
- B. A random 80/20 split after removing the 24 resubmitted copies.
- C. Train on tickets before 2025-09-01 and test on tickets from that date, after removing the
  resubmitted copies.
- D. Evaluate on the same tickets used to choose the few-shot examples, to maximise the data.

<details><summary>Answer</summary>

**C.** It mimics production: everything you built with comes from the past, and the test comes from
the future, drift included. In the lab the same prompt scored 0.959 with A, 0.921 with B and 0.828
with C. B removes the worst duplicates but still mixes past and future. D measures memory, not
routing.

</details>

### Q2

Spot the bug in this prompt builder.

```python
def user_message(ticket: Ticket) -> str:
    context = (f"language={ticket.language}; product={ticket.product}; "
               f"resolution={ticket.resolution_code}")
    return f"Customer context: {context}\nTicket: {clean_text(ticket.body)}"
```

<details><summary>Answer</summary>

`resolution_code` is written by an agent after the ticket is solved and maps one-to-one to the intent.
It does not exist when a new ticket arrives, so putting it in the prompt is target leakage. In the lab
it raised accuracy from 0.828 to 0.920, a gain that would vanish on the first real ticket. Build the
context from an allow-list of fields known at prediction time.

</details>

### Q3

In the lab, with bilingual label definitions, which statement about few-shot examples is true?

- A. Eight random examples beat eight similar examples because they cover more intents.
- B. Eight similar examples scored 0.828 against 0.728 for eight random ones.
- C. Examples never help once the label definitions are good.
- D. One similar example is always as good as eight.

<details><summary>Answer</summary>

**B.** Similar examples share words with the ticket and give the toy learner real evidence; random
examples rarely do. A is wrong in this lab (balanced examples, one per intent, reached only 0.740). C
is contradicted by the jump from 0.723 to 0.828. D is wrong too: one similar example scored 0.782.

</details>

### Q4

A teammate adds "Think step by step" to the routing prompt. On the test set, accuracy stays at 0.828
and mean completion tokens go from 9.4 to 35.7. What do you do?

<details><summary>Answer</summary>

Remove it for this task, or at least do not ship it without a gain on a real model. Chain-of-thought
helps multi-step problems; routing is a one-step recognition task. The change multiplied the output
tokens, which are usually the slowest and most expensive part of a request, for no accuracy gain. Keep
the "Answer:" parsing in case you need step-by-step prompting for a multi-step task later.

</details>

### Q5

Spot the bug.

```python
def system_message(request_id: str) -> Message:
    header = f"Request {request_id} received at {datetime.now().isoformat()}\n"
    return Message("system", header + ROUTER_INSTRUCTIONS)
```

<details><summary>Answer</summary>

The first line changes on every request, so no two prompts share a prefix and prefix caching never
hits. In the lab's simulation the stable system prompt had 399 hits out of 400 and 0.496 of prompt
tokens served from cache; with a request ID and date first it had 0 hits. Put variable data at the end
of the context or in request metadata.

</details>

### Q6

You have eight examples ranked by similarity. Where should the most similar one go?

- A. First, right after the system message.
- B. In the middle, so it is surrounded by context.
- C. Last, right before the ticket.
- D. Anywhere: models ignore position.

<details><summary>Answer</summary>

**C.** Studies such as "Lost in the Middle" found that models use information at the start or end of
a long context more reliably than information in the middle. The lab's assembler puts the stable
system prompt first and the most relevant example next to the ticket. D is wrong for many real models,
although the lab's toy is order-blind by default; that is why you test position with a probe.

</details>

### Q7

At a 700-token budget, removing near-duplicate examples saves 102.1 tokens per ticket but accuracy
drops from 0.843 to 0.830. Your manager asks whether to keep deduplication. What do you answer?

<details><summary>Answer</summary>

It is a trade-off to decide with numbers, not a rule. Deduplication cut cost by about 15% of the
prompt (583.0 against 685.1 tokens) and lost 1.3 points, because duplicate examples were acting as
extra votes for the right label. Rerun the comparison on the real model, then weigh the token saving
at your volume against the extra misrouted tickets. A middle option is to keep the budget and
deduplicate only exact copies.

</details>

### Q8

Calculate the LoRA size for one weight matrix with 4,096 outputs and 1,024 inputs at rank 16. What
share of the full matrix is that?

<details><summary>Answer</summary>

`r · (d_in + d_out) = 16 · (1,024 + 4,096) = 81,920` trainable numbers. The full matrix has
`1,024 · 4,096 = 4,194,304`, so LoRA trains a share of 0.01953, about 2%. Check it with
`uv run lab lora-params --d-in 1024 --d-out 4096`.

</details>

### Q9

A model has 13 billion parameters. How much memory do its weights alone need in 16-bit and in 4-bit
precision, and which technique trains adapters on top of the 4-bit version?

<details><summary>Answer</summary>

16-bit: `13 × 10⁹ × 2 bytes = 26.0 GB`. 4-bit: `13 × 10⁹ × 0.5 bytes = 6.5 GB`. QLoRA keeps the frozen
base in 4-bit and trains 16-bit LoRA adapters on top. Training needs extra memory for activations
and for the adapters' gradients and optimizer state, so these numbers are a lower bound.

</details>

### Q10

Why does LoRA initialise `B` to zero?

- A. To make the adapter smaller on disk.
- B. So that the adapted model starts exactly equal to the pre-trained model.
- C. To make the update full rank.
- D. Because A and B must always be equal.

<details><summary>Answer</summary>

**B.** With `B = 0` the update `(alpha/r)·B·A` is zero, so training starts from the base model's
behaviour and only moves away as far as the data justifies. A is random so that the gradient for B is
not zero. C and D are false: the update has rank at most r, and A and B have different shapes.

</details>

### Q11

After 4,000 training steps on the September tickets, a fully fine-tuned router scores 0.855 on
October–December and 0.792 on August. The frozen base scored 0.828 and 0.917. A teammate wants to ship
it because "new tickets improved". What do you do?

<details><summary>Answer</summary>

Do not ship. The model forgot how to route old tickets (catastrophic forgetting) and over-fitted the
97 September tickets: after only 100 steps it scored 0.927 on new and 0.906 on old. Use the old-data
set as a release gate, train less (or stop early), and mix old tickets into training: replay of 100
old tickets kept August at 0.865 even at 4,000 steps. Do not assume LoRA solves it; LoRA rank 4 also
fell, to 0.812.

</details>

### Q12 🔁 (module 03)

The playground conversation has grown past the model's context window. What is the best first
strategy?

- A. Drop the system prompt, since the model has seen it already.
- B. Keep the system prompt and the latest turns verbatim, and replace older turns with a summary.
- C. Send everything and let the provider truncate.
- D. Silently start a new conversation.

<details><summary>Answer</summary>

**B.** The model is stateless; your system decides what it sees each turn. Keep what is required (the
system prompt), what is most relevant (recent turns) and a compact version of the rest. Lesson 4.3
generalises this into a budgeted assembler with priorities and a drop report. A loses the rules, C
cuts by position instead of importance, and D surprises the user.

</details>

### Q13 🔁 (module 02)

Which decoding setting fits a router whose output must be one of eight labels?

- A. Temperature 1.0, for more diverse answers.
- B. Temperature 0 (greedy decoding), for repeatable answers.
- C. A high top-p, so that rare labels get a chance.
- D. A large `max_tokens`, so that the model can explain itself.

<details><summary>Answer</summary>

**B.** Routing needs the single most likely label every time, and the same ticket should get the
same route. The lab's `classify` calls the model with `temperature=0.0` and a small `max_tokens`. A
and C add randomness that only creates inconsistent routes; D invites long outputs that cost tokens
and complicate parsing.

</details>

### Q14 🔁 (module 02)

How does LoRA relate to supervised fine-tuning (SFT)?

- A. LoRA replaces the need for labelled examples.
- B. LoRA is a way to run SFT (or other training) while updating only small low-rank adapters.
- C. LoRA is a decoding method used after SFT.
- D. LoRA is a reward model for preference optimisation.

<details><summary>Answer</summary>

**B.** SFT defines *what* you train on (input and desired output pairs, with the loss on the answer);
LoRA defines *which* parameters change (a low-rank update instead of all weights). You still need good
labelled data, and data quality problems such as duplicates or wrong labels hurt LoRA just as much.

</details>

## Scoring guide

Give yourself one point per question.

| Score | What it means | What to do |
|---|---|---|
| 12–14 | You can apply the module's ideas to new cases | Move on to module 05 |
| 9–11 | Solid, with gaps | Reread the lesson section behind each miss and rerun the matching lab step |
| 0–8 | The ideas have not settled yet | Redo lab steps 2, 4, 6 and 9, then retake the quiz in a few days |

## ⏭️ Next

[Module 05 — RAG I: Retrieval →](../05-rag-retrieval/README.md)
