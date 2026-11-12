🌐 English · [Español](cheatsheet.es.md)

# Module 02 cheatsheet — From base model to assistant

## The path

```text
base model → SFT (format, tone, stopping) → preference optimisation (RLHF with PPO, or DPO) → assistant
every step: decoding turns logits into text · evaluation decides whether it worked
```

## Decoding

| Knob | Rule | Use it for |
|---|---|---|
| Temperature T | softmax(logits / T); T < 1 sharpens, T > 1 flattens | Creativity versus stability |
| Greedy | argmax at every step; deterministic | Extraction, classification, JSON |
| Beam search (k beams) | Keep the k best partial sequences by summed log-probability | One-right-answer tasks (translation); bland on open-ended text |
| Top-k | Keep the k most likely tokens | Simple tail cut; does not adapt |
| Top-p (nucleus) | Keep the smallest set with cumulative probability ≥ p | Adaptive tail cut |
| Repetition penalty | Seen token: positive logit / penalty, negative logit × penalty | Breaking loops |
| Frequency / presence | Subtract α × count, or β once present | API-style loop control |
| Stop sequences | End when the text contains the string (not included) | Clean endings |
| Max tokens | Hard ceiling; `finish_reason = "length"` | Cost and latency guard, not a fix for rambling |
| Seed | Same seed and settings give the same sample (best effort in production) | Reproducibility |

Processor order in the lab: penalties → top-k → top-p → softmax at T → draw.

Why models ramble: no learned end-of-turn (fix: SFT), no stop sequence, max tokens hit, repetition loops.

## Supervised fine-tuning

- Chat template: `<|role|>\ncontent<|end|>\n` per turn; generation prompt = open `<|assistant|>\n`.
- Loss mask: 1 on assistant tokens **and** their `<|end|>`, 0 on system, user and role markers.
- SFT teaches format, tone, habits and stopping; it is a poor way to teach changing facts.
- Data audit: empty, low quality, wrong language, PII, eval contamination, outdated facts, duplicates.
- Precision = correct flags / flags; recall = correct flags / labelled rows. Read the disagreements.

## Preference optimisation

| Idea | Formula |
|---|---|
| Bradley–Terry | P(A over B) = σ(r(A) − r(B)); loss = −log σ(r(chosen) − r(rejected)) |
| PPO-style reward | r(x, y) − β · (log π(y given x) − log π_ref(y given x)) |
| PPO clip | min(ρ · A, clip(ρ, 1 − ε, 1 + ε) · A), with ρ = π_new / π_old |
| DPO loss | −log σ(β · [(log π(c) − log π_ref(c)) − (log π(r) − log π_ref(r))]); equals ln 2 at the start |
| Implicit reward (DPO) | β · (log π(y) − log π_ref(y)) |

- Verifiable rewards: a program checks the answer (maths, tests, schemas). Test the verifier for loopholes.
- Evaluate reward models on **new prompts** and against "longer answer wins".
- Reward hacking: the policy exploits the reward (padding, flattery). Watch length, KL and samples.

## Evaluation

| Method | Measures | Main trap |
|---|---|---|
| EM / token F1 | Normalised string or word overlap | Correct paraphrases score low |
| BLEU / ROUGE-L | n-gram precision / longest common subsequence | A wrong answer with the right words scores high |
| Perplexity | exp(mean negative log-likelihood) on held-out text | Predictable ≠ good; tokenizer-dependent |
| Benchmarks (MMLU, GSM8K, HumanEval) | Fixed tasks with automatic scoring | Contamination; may not match your task |
| Pairwise human votes | Which answer people prefer | Needs many votes; report intervals |
| Elo | Online rating updates (K × (result − expected)) | Depends on vote order |
| Bradley–Terry ratings | One strength per model from all votes | Needs enough votes per pair |
| LLM-as-a-judge | Cheap pairwise or graded verdicts | Position, verbosity and self-preference bias |

Contamination check: share of each eval question's n-grams found in a document ≥ threshold → flag, then read.

## Lab commands

```text
uv run lab corpus | next-token --prompt "..." | generate [--strategy greedy|beam|sample] [--temperature T]
           [--top-k K] [--top-p P] [--repetition-penalty R] [--stop S] [--max-tokens N] [--seed N]
uv run lab decode-bench      # 2.1 diversity vs likelihood table
uv run lab chat-template     # 2.2 template and loss mask
uv run lab sft-clean         # 2.2 data audit against labels
uv run lab reward            # 2.3 reward models and the length trap
uv run lab verify | dpo      # 2.3 verifiable reward, DPO loss, KL-penalised reward
uv run lab metrics           # 2.4 EM, F1, BLEU, ROUGE-L, perplexity
uv run lab contamination     # 2.4 eval questions in the web crawl
uv run lab arena | judge     # 2.4 Elo vs Bradley–Terry, judge swap test
```

## Three most common mistakes

1. **Shrinking max tokens to stop rambling.** It cuts sentences; fix the stop condition or the model.
2. **Trusting a perfect held-out score.** Check whether prompts or answers leaked across the split.
3. **Asking an LLM judge once, in a fixed order.** Swap positions, watch length, calibrate on humans.

---

[← Previous: Quiz](quiz.md) · [🏠 Module 02](README.md) · [Next: Module 03 →](../03-llm-playground/README.md)
