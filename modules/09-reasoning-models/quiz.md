🌐 English · [Español](quiz.es.md)

# Module 09 — Quiz

> 13 questions, about 25 minutes. Answer on paper first, then open the answers. Questions marked 🔁
> review earlier modules.

### Q1 — Thinking tokens

A reasoning API returns a short answer and hides its reasoning. Which statement is true?

- A. Hidden reasoning tokens are free; only visible tokens are billed.
- B. Hidden reasoning tokens are generated output tokens: they cost money and time, and the usage data reports them.
- C. Hidden reasoning tokens are counted as prompt tokens.
- D. Reasoning tokens only exist when you set a high reasoning effort.

<details><summary>Answer</summary>

**B.** Thinking tokens are generated like any output. You pay for them and wait for them even when
you never see them, so budget from the usage data, not from the length of the visible answer.

</details>

### Q2 — Price of a point

In `uv run lab solve`, chain of thought scores 49.4% with 72.7 tokens per answer, and thinking at
high effort scores 71.9% with 194.1 tokens per answer. How many extra tokens per answer does each
extra percentage point cost?

<details><summary>Answer</summary>

(194.1 − 72.7) / (71.9 − 49.4) = 121.4 / 22.5 ≈ **5.4 tokens per point**. Averages hide the split:
on the five-step hub problems the same switch gains 40 points, on two-step problems 13.2.

</details>

### Q3 — The plateau

Majority vote over the numeric problems reaches 89.1% at N = 15 and is still 89.1% at N = 45, while
pass@15 is already 100%. What is the best explanation?

- A. 45 samples are too few; at N = 200 the vote would reach 100%.
- B. The vote is wrong on problems where the same wrong answer (a trap) is the most frequent one, and more samples make that answer win more reliably.
- C. The tie-breaking rule is wrong.
- D. pass@N is computed incorrectly.

<details><summary>Answer</summary>

**B.** All 12 problems the vote gets wrong at N = 45 have a trap answer as the plurality. Voting
removes independent errors and amplifies correlated ones. A is the tempting wrong answer: more
samples only sharpen the estimate of the most frequent answer, which is wrong on those problems.

</details>

### Q4 — Spot the bug

```python
def vote(answers: list[str]) -> str:
    return Counter(answers).most_common(1)[0][0]

vote(["175.1", "175.10", "90"])
```

Why can this vote return the wrong answer for an order total of €175.10?

<details><summary>Answer</summary>

It counts raw strings, so `175.1` and `175.10` are two different answers with one vote each, tied
with `90`; whichever comes first wins. Normalise answers before counting (the lab's `answer_key`
turns both into the same exact fraction), and make the tie-breaking rule explicit.

</details>

### Q5 — pass@k by hand

You drew n = 5 samples for a problem and c = 1 is correct. Using the unbiased estimator
pass@k = 1 − C(n − c, k) / C(n, k), what is pass@2?

<details><summary>Answer</summary>

C(4, 2) = 6 and C(5, 2) = 10, so pass@2 = 1 − 6/10 = **0.4**. Intuition: of the 10 ways to pick 2 of
the 5 samples, 4 include the correct one.

</details>

### Q6 — ORM or PRM

On the 4,950 numeric samples, the simulated ORM lets through 571 wrong answers and rejects 518 right
ones; the simulated PRM lets through 64 and rejects 906. You want to rank 9 samples of the same
question and keep one. Which do you prefer, and why might you prefer the other as a filter on a
stream of single answers?

<details><summary>Answer</summary>

For ranking, the PRM: it can see misreadings, and best-of-9 with the PRM reached 95.5% against 86.4%
with the ORM. Ranking only needs the right answer to score above the wrong ones *for the same
question*. As a pass/fail filter on single answers, the PRM's 906 false rejects would throw away a lot
of good work; you would tune its threshold or combine it with an exact check.

</details>

### Q7 — Group-relative advantages

A GRPO group of four answers gets rewards 1, 0, 0, 0. Compute the advantages (population standard
deviation).

<details><summary>Answer</summary>

Mean = 0.25. Standard deviation = √((0.75² + 3 × 0.25²) / 4) = √0.1875 ≈ 0.433. Advantages:
0.75 / 0.433 ≈ **1.73** for the right answer and −0.25 / 0.433 ≈ **−0.58** for each wrong one. They
sum to zero. With rewards 1, 1, 1, 1 every advantage would be 0 and the group would teach nothing.

</details>

### Q8 — What would you do: puzzles on a budget

Jordan needs the number puzzles solved, and Sam's limit is about 500 prompt tokens and 7 calls per
puzzle. From the lab: sampling until verified with up to 15 tries solves 41/50 (485.7 prompt tokens,
6.2 calls); ToT BFS with breadth 5 solves 36/50 (2,381.9 prompt tokens, 28.6 calls). Which do you
ship?

<details><summary>Answer</summary>

Sampling until verified. It fits both limits and solves more puzzles, because the exact verifier
makes every extra sample safe. ToT wins on generated tokens, but every propose and evaluate call
resends the puzzle, so it blows the prompt-token and call budgets.

</details>

### Q9 — Spot the bug: a suspicious STaR result

A teammate's STaR loop reports a far bigger jump after two rounds than anything in the lab table.
Their evaluation calls `evaluate(config, train)`. What is wrong?

<details><summary>Answer</summary>

They evaluate on the training problems, whose rationales the model was just fine-tuned on: data
leakage. Evaluate on held-out problems that never enter the loop, as the lab does with the even ids.
The honest held-out number in the lab is 72.5% after four rounds.

</details>

### Q10 — What would you do: signs of reward hacking

After switching best-of-N to a new learned verifier, the average verifier score of the chosen answers
goes up every week, the chosen answers get longer, and the spot-check accuracy measured by people is
flat. What do you suspect, and what do you do?

<details><summary>Answer</summary>

Reward hacking: the selection is rewarding something the verifier likes (here, length) rather than
correctness, as the length-biased ORM in lesson 9.3 did. Measure against ground truth on a held-out
set, look for the bias (score versus length), add exact checks where possible, and fix or replace the
verifier before relying on its scores.

</details>

### Q11 🔁 (module 02)

You run self-consistency with 9 samples per question, but every sample is identical and the vote
never helps. Which decoding setting from module 02 is the likely cause, and why?

<details><summary>Answer</summary>

Greedy decoding (temperature 0): the model always picks the most likely token, so every sample is
the same chain and the vote has nothing to choose between. Self-consistency needs diverse samples:
use a temperature above zero (often with top-p), so different chains, and different mistakes, appear.

</details>

### Q12 🔁 (module 04)

Jordan approves distilling verified reasoning traces into a small open model, on a single modest GPU.
Which fine-tuning approach from module 04 fits, and what does it change in the model?

<details><summary>Answer</summary>

A parameter-efficient method such as LoRA (or QLoRA, which also quantises the frozen base model). It
freezes the original weights and trains small low-rank matrices added to some layers, so memory and
compute needs drop sharply and the adapter is small enough to store and swap. The training data is
still ordinary SFT data: question, reasoning trace, answer.

</details>

### Q13 🔁 (module 08)

In module 08 you used tree search for a web agent. Name one thing that makes tree search harder for
an agent acting on the web than Tree of Thoughts on the target puzzles.

<details><summary>Answer</summary>

Any of these: actions can have side effects and cannot always be undone, so backtracking is not free;
the state is a whole web page or conversation, not a short list of numbers, so it is expensive to
copy and to evaluate; and there is rarely an exact verifier for the final answer, so the search is
guided only by learned or heuristic judgements.

</details>

## 📊 Scoring guide

| Score | What it means |
|---|---|
| 11–13 | Ready for module 10. |
| 8–10 | Re-read the self-checks of the lessons where you missed questions, then continue. |
| 0–7 | Work through the lab again, especially steps 3 to 7, before moving on. |

## ⏭️ Next

[Module 10 — Deep Research and Local Deployment](../10-deep-research-and-local-deployment/README.md)
