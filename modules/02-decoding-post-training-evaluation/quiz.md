🌐 English · [Español](quiz.es.md)

# Module 02 quiz — From base model to assistant

Answer each question before opening the solution. Write your answers down; explaining *why* counts as
much as the letter or the number. Questions marked 🔁 revisit module 01.

### Q1

You lower the sampling temperature from 1.0 to 0.5. What happens to the next-token distribution?

- A. Every token becomes equally likely
- B. The most likely tokens get more of the probability and the tail gets less
- C. Random noise is added to the logits
- D. Nothing changes until the temperature reaches 0

<details><summary>Answer</summary>

**B.** Temperature divides the logits before softmax; dividing by 0.5 doubles the gaps between scores.
In the lab, P("device") after "to reset the" goes from 0.612 at T = 1.0 to 0.996 at T = 0.5. Temperature
does not add noise (C), and a high temperature flattens the distribution but never makes it exactly
uniform (A).

</details>

### Q2

The next-token probabilities, sorted, are 0.5, 0.3, 0.15 and 0.05. How many tokens does top-p keep with
p = 0.8? And with p = 0.81?

<details><summary>Answer</summary>

**2 and 3.** The cumulative sums are 0.5, 0.8, 0.95 and 1.0. Top-p keeps the smallest prefix whose sum
reaches p: 0.8 is reached after two tokens; 0.81 needs the third. The lab's
`test_top_p_keeps_smallest_set_reaching_p` checks exactly these cases.

</details>

### Q3

For which task is beam search the most sensible choice?

- A. A friendly chatbot answering open questions
- B. Brainstorming product names
- C. Translating a short product label from English to Spanish
- D. Writing varied marketing emails

<details><summary>Answer</summary>

**C.** Beam search looks for the single most likely output, which suits tasks with one correct answer.
On open-ended prompts it produces short, generic text: in the lab, "to reset the" becomes "device keeps
working during the holidays." and the benchmark shows only 6 different outputs out of 30.

</details>

### Q4

In production, 30% of the assistant's responses end with `finish_reason = "length"`. What do you do
first?

<details><summary>Answer</summary>

Read a sample of those responses. Most of them are cut mid-sentence, which users see as broken answers.
Then find the cause: answers got longer after a prompt change, the model stopped emitting its end-of-turn
token, a loop, or a max-token limit that is simply too low. Fix the cause (stop sequences, prompt,
repetition penalty, model), and keep the share of `length` endings on a dashboard. Lowering max tokens
further would make it worse.

</details>

### Q5 — spot the bug

```python
def loss_mask(tokens):
    mask, in_assistant = [], False
    for tok in tokens:
        if tok in ("<|system|>", "<|user|>", "<|assistant|>"):
            in_assistant = tok == "<|assistant|>"
            mask.append(0)
        elif tok == "<|end|>":
            mask.append(0)
            in_assistant = False
        else:
            mask.append(1)
    return mask
```

<details><summary>Answer</summary>

Two bugs. The `else` branch appends 1 for every ordinary token, so the system prompt and the user's
message are trained too; it should append `1 if in_assistant else 0`. And the end marker of an assistant
turn gets 0, so the model never learns to stop; it should get `1 if in_assistant else 0`.

</details>

### Q6

The SFT cleaner flags 11 rows as contaminated with the evaluation set, but the labels mark only 5. What
explains the other 6?

- A. The n-gram threshold is too low, so these are false alarms
- B. They copy the reference answer of evaluation question q027 word for word, a real leak the labels missed
- C. They are Spanish rows that the language check mislabelled
- D. They are exact duplicates of the 5 labelled rows

<details><summary>Answer</summary>

**B.** All 6 contain "In the app under Account > Orders: tap the order and choose Download invoice.",
the reference answer of q027. Precision of 0.45 against the labels hides a correct detection. Read the
disagreements before trusting either the check or the labels.

</details>

### Q7

A Bradley–Terry reward model scores answer A at r = 1.2 and answer B at r = 0.2. What probability does
it give to people preferring A?

<details><summary>Answer</summary>

**About 0.731.** P(A over B) = σ(1.2 − 0.2) = σ(1.0) = 1 / (1 + e^(−1)) ≈ 0.731. Only the difference
matters: 5.2 versus 4.2 would give the same probability.

</details>

### Q8

A words-plus-length reward model scores 1.000 on the training pairs and 1.000 on the official test split.
What is the most likely explanation?

- A. The model has learned the agents' preferences perfectly
- B. The same questions appear in both splits, so the model memorised which answer goes with which question
- C. The test split is too easy because it has fewer pairs
- D. Length bias cancels out on the test split

<details><summary>Answer</summary>

**B.** The 23 questions repeat across the splits. On a prompt-disjoint split the same kind of model drops
to 0.714, and its largest weights include the length feature. Perfect held-out scores are a reason to look
for leakage.

</details>

### Q9

In PPO-based RLHF, the reward model gives r = 1.5, the policy's log-probability of the response is
−10.0, the reference model's is −14.0, and β = 0.2. What reward does PPO optimise for this sample?

<details><summary>Answer</summary>

**0.7.** r − β · (log π − log π_ref) = 1.5 − 0.2 × (−10.0 − (−14.0)) = 1.5 − 0.8 = 0.7. The policy has
drifted 4 nats towards this response, and the KL penalty charges for that drift.

</details>

### Q10

At the start of DPO training the policy equals the reference model. What is the DPO loss of every pair,
and why?

<details><summary>Answer</summary>

**ln 2 ≈ 0.693.** The margin (policy log-ratio of chosen minus that of rejected) is 0, and
−log σ(0) = −log 0.5 = ln 2. Training then lowers the loss by widening the margin, which can even happen
while the chosen answer's probability falls, as long as the rejected one falls further.

</details>

### Q11

The reference answer is "30 days from delivery." Which candidate gets the highest ROUGE-L?

- A. "You have 30 days after delivery to return it."
- B. "Thirty days, counted from the day it arrives."
- C. "14 days from delivery."
- D. "Returns are accepted for 30 days from delivery, opened or not."

<details><summary>Answer</summary>

**C**, the only wrong answer, with 0.80 in the lab's metric table, because it shares three of its four
words with the reference. Overlap metrics measure wording, not correctness; that is why rag_eval also
lists `must_not_contain` strings such as "14 days".

</details>

### Q12 — what would you do

Jordan wants to use an LLM judge to compare two vendors' models every week, asking it once per pair
"Which answer is better, A or B?" with vendor X always shown as A. What do you change?

<details><summary>Answer</summary>

Randomise or swap the order and ask twice per pair, counting only consistent verdicts, because judges
have position bias (the lab's judge agrees with humans 1.000 of the time with the better answer first
and 0.667 with it second). Track how often the longer answer wins (verbosity bias), use a judge from a
different family than both vendors (self-preference), calibrate it on human-labelled Hearthwise pairs,
and report intervals, not just win rates.

</details>

### Q13 🔁 (module 01)

Two pages on studyhub.example are clean, well-written English with no spam and no personal data. Why
should a pre-training cleaning pipeline still drop them?

<details><summary>Answer</summary>

They copy evaluation material: 6 rag_eval questions with answers and 6 reasoning problems with
solutions. Training on them would let a model recall the answers and inflate its scores
(contamination). Their `keep` label in `datasets/web/labels.jsonl` is false for that reason, and the
contamination detector in lesson 2.4 finds both.

</details>

### Q14 🔁 (module 01)

In the web crawl, labelled near-duplicate pairs have a word 5-gram Jaccard similarity of at least 0.969,
and two different kept pages never exceed 0.396. Which threshold would you choose for near-duplicate
removal on this crawl, and what is the risk of choosing 0.3?

<details><summary>Answer</summary>

Anything comfortably between the two, for example 0.8, separates them on this crawl. At 0.3 you would
also remove distinct pages that only share common phrasing (two kept pages reach 0.396), losing good
data. The same precision–recall reasoning applies to the SFT duplicate filter in lesson 2.2.

</details>

### Q15 🔁 (module 01)

Model X reports a perplexity of 8 on your support emails, model Y a perplexity of 12. Can you conclude
that X predicts the emails better?

<details><summary>Answer</summary>

Not unless they use the same tokenizer. Perplexity is per token, and a tokenizer that splits text into
more, smaller pieces makes each prediction easier. Compare on the same tokenization, or normalise per
character or per byte. In the lab, Spanish pages look only moderately surprising (20.30) because 80% of
their words collapse into `<unk>`: the vocabulary shapes the number.

</details>

## Scoring guide

| Correct answers | What it means |
|---|---|
| 13–15 | You are ready for module 03. |
| 10–12 | Good. Reread the self-check of the lessons behind the questions you missed. |
| 7–9 | Redo the related lab steps and the Break-it items before moving on. |
| 0–6 | Go back through the lessons in order; the lab outputs are the best anchor. |

## ⏭️ Next

[Module 03 — Build an LLM Playground →](../03-llm-playground/README.md)

---

[← Previous: Lab](lab/README.md) · [🏠 Module 02](README.md) · [Next: Cheatsheet →](cheatsheet.md)
