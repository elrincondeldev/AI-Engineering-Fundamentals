🌐 English · [Español](cheatsheet.es.md)

# Module 09 — Cheatsheet: reasoning and thinking models

## Key terms

| Term | One-line definition |
|---|---|
| Chain of thought (CoT) | Intermediate steps written before the answer; each step becomes input for the next tokens |
| Reasoning (thinking) model | A model trained to reason at length (plan, check, backtrack) before answering |
| Thinking tokens | Output tokens spent on reasoning; billed and slow even when hidden |
| Reasoning effort | A setting that trades thinking tokens for accuracy, with diminishing returns |
| Inference-time scaling | Spending more compute per question: parallel samples or sequential revisions |
| Self-consistency | Sample several chains, return the most frequent (normalised) answer |
| Best-of-N | Sample N answers, keep the one a verifier scores highest |
| Verifier | Program or model that judges an answer or a step |
| ORM / PRM | Outcome reward model (scores a whole solution) / process reward model (scores each step) |
| Tree of Thoughts (ToT) | Model proposes moves and rates states; code runs BFS or DFS over them |
| Reward hacking | Optimising a proxy score rewards its biases instead of the goal |
| STaR | Generate, keep correct rationales, rationalise failures with a hint, fine-tune, repeat |
| GRPO | RL that scores each sample against its own group; no value model |
| Distillation | SFT of a small model on verified traces from a stronger model |
| Meta-CoT | Training on linearised search traces so search happens inside one chain |

## Formulas

```text
pass@k (unbiased, n samples, c correct)  = 1 - C(n - c, k) / C(n, k)
majority@k                               = accuracy of the most frequent normalised answer among k
best-of-N                                = argmax_i verifier_score(sample_i)
GRPO advantage of sample i               = (r_i - mean(r)) / std(r)      (0 for all if std = 0)
PRM chain score (this lab)               = min over steps of step_score
cost per question                        = prompt tokens + generated tokens (thinking included)
```

## Which mistake, which fix

| Error type | Example | Caught by | Not caught by |
|---|---|---|---|
| Slip (random) | `6 x 7 = 52` | voting, arithmetic re-check, self-checks | nothing special needed |
| Trap (systematic) | counting Wi-Fi bulbs on the hub | re-reading, PRM, targeted rule | majority vote, arithmetic re-check |
| Unfaithful chain | right answer, wrong step | step checks, PRM | answer-only filters, ORM |

## Choosing a method

| Situation | Start with |
|---|---|
| Short question, cheap mistakes | Plain CoT, low effort |
| Long multi-step question | Thinking mode or CoT + checker; raise effort only where it pays |
| An exact checker exists | Sample until verified (cheap, safe, keeps improving with budget) |
| No checker, many samples affordable | Best-of-N with a reward model; watch for bias |
| Need fewer generated tokens, calls are cheap | Step-level search (ToT, PRM beam search) |
| Need a small local reasoning model | Distil verified traces, then optional RL with rule-based rewards |

## Headline numbers from the lab (simulated reasoner)

| Result | Value |
|---|---|
| direct / CoT / think-high accuracy (160 problems) | 33.1% / 49.4% / 71.9% |
| tokens per answer, CoT vs think-high | 72.7 vs 194.1 |
| majority vote at N = 15 and N = 45 (110 numeric problems) | 89.1% and 89.1% |
| best-of-9 with ORM / PRM | 86.4% / 95.5% |
| target puzzles: CoT / sample until verified (15) / ToT BFS breadth 5 | 11 / 41 / 36 of 50 |
| STaR test accuracy, round 0 to round 4 | 51.2% to 72.5% |
| RL toy: tokens per answer, step 0 to step 8 | 77.9 to 113.0 |

## Lab commands

```bash
cd modules/09-reasoning-models/lab && uv sync && uv run pytest
uv run lab solve [--id r076] [--mode think-high]   # direct vs CoT vs thinking
uv run lab scale                                   # vote, best-of-N, pass@N vs tokens
uv run lab revise                                  # self-critique vs re-check vs PRM feedback
uv run lab search [--id r076]                      # verifiers, PRM beam search, reward hacking
uv run lab tot [--id r113]                         # Tree of Thoughts vs sampling until verified
uv run lab star [--no-rationalize] [--strict]      # STaR loop
uv run lab rl [--steps 8] [--group 8]              # GRPO-style toy
```

## The three most common mistakes

1. **Trusting agreement.** Many samples giving the same answer is not proof; correlated errors (traps) win votes.
2. **Counting the wrong tokens.** Hidden thinking tokens and resent prompt tokens are both on the bill.
3. **Optimising the verifier instead of the goal.** Always measure against ground truth the verifier never sees.
