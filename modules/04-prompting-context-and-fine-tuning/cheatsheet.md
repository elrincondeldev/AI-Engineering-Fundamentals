🌐 English · [Español](cheatsheet.es.md)

# Module 04 cheatsheet — Adapting LLMs

## Key terms

| Term | Meaning |
|---|---|
| Zero-shot / few-shot | Prompt with no solved examples / with k solved examples (in-context learning) |
| Role-specific prompting | A system message that sets who the model is, its scope and its rules |
| User-context prompting | Facts about the customer known at prediction time (language, product, channel) |
| Chain-of-thought (CoT) | Ask for intermediate reasoning before the answer; for multi-step problems |
| Context engineering | Choosing which pieces enter the window, their size, order and priority |
| Compaction | Clean, deduplicate, summarise, truncate |
| Prefix caching | Reuse of the provider's work for a byte-identical prompt prefix |
| Context rot | Quality falling as the window fills with stale, redundant or marginal material |
| PEFT | Parameter-efficient fine-tuning: freeze the base, train a small add-on |
| LoRA / QLoRA | Low-rank update of frozen weights / the same on top of a 4-bit base |
| Target leakage | A feature that is only known after the answer exists (`resolution_code`) |
| Concept drift | The input-label relationship changes (Lookout C2 launch, 2025-09-01) |
| Catastrophic forgetting | Losing old skills after training on new, narrow data |

## Formulas

```text
LoRA update        W' = W + (alpha / r) · B · A        B: d_out × r (zeros at start), A: r × d_in (random)
LoRA size          r · (d_in + d_out)                  full matrix: d_in · d_out
Share of full      r · (d_in + d_out) / (d_in · d_out)  4096 × 4096, r = 8 -> 65,536 = 0.00391
LoRA gradients     grad_W = dLoss/dW'   grad_B = (alpha/r) · grad_W · Aᵀ   grad_A = (alpha/r) · Bᵀ · grad_W
Weights memory     params × bits / 8 bytes             7e9 params: 14.0 GB at 16-bit, 3.5 GB at 4-bit
Jaccard            |A ∩ B| / |A ∪ B|
MMR score          (1 - λ) · sim(query, c) - λ · max sim(c, chosen)
```

## Choosing a technique

| Question | Prompt / context / RAG | Fine-tune (PEFT) |
|---|---|---|
| Labelled data? | Few or none | Hundreds+, clean |
| Where to pay? | Per request (tokens) | Up front (training) |
| Latency budget? | Moderate | Tight (shorter prompts) |
| How often does it change? | Often | Rarely |
| What must be controlled? | Readable, auditable logic | Format/skill a prompt cannot hold |

Order of attempts: prompt + honest eval → context / retrieval → LoRA → (rarely) full fine-tuning.

## Lab results to remember (temporal test, 400 tickets)

| Result | Value |
|---|---|
| Same prompt, random split vs temporal split | 0.959 vs 0.828 |
| `resolution_code` alone / pasted into the prompt | 1.000 / 0.920 |
| Label names only → bilingual definitions (0-shot) | 0.270 → 0.723 |
| 8 random vs 8 similar examples | 0.728 vs 0.828 |
| + user context (language, product, channel) | 0.845, 790.5 prompt tokens |
| Step-by-step: accuracy, completion tokens | 0.828, 9.4 → 35.7 |
| Valid JSON at first try (strict / lenient), after 1 retry | 379 / 390 of 400, 0 invalid |
| Quoted-reply tickets, 8-shot raw vs cleaned | 0.596 vs 0.830 |
| Prefix cache hit rate, stable vs request ID first | 0.998 vs 0.000 |
| Oct–Dec: refreshed prompt / LoRA r=4 / full fine-tuning | 0.901 / 0.905 / 0.927 |
| August after 4,000 steps: full / LoRA r=4 / full + replay | 0.792 / 0.812 / 0.865 |

## Context assembly checklist

1. Required first: system prompt and query; never drop them.
2. Optional pieces by priority until the budget is full; log every drop with its reason.
3. Order: stable prefix → examples (least to most relevant) → profile → query.
4. Clean before anything else (quotes, signatures, HTML entities, PII).
5. Measure deduplication and summarisation; do not assume they help.
6. Keep anything that varies (IDs, dates) out of the top of the prompt.

## Lab commands

```bash
cd modules/04-prompting-context-and-fine-tuning/lab && uv sync && uv run pytest
uv run lab audit                 # duplicates, PII, leak audit, drift
uv run lab splits                # random vs temporal split, leak through the prompt
uv run lab show-prompt --k 2     # the exact messages for one ticket (--ticket T00843)
uv run lab prompts               # zero/few-shot, selection, user context
uv run lab format                # JSON validation, retries, step-by-step cost
uv run lab context --budget 500  # one assembled prompt and its drop report
uv run lab budget                # budgets x compaction policies, too many examples
uv run lab cache                 # prefix cache simulation
uv run lab drift                 # adapt by rewriting definitions or refreshing the pool
uv run lab lora                  # LoRA ranks vs full fine-tuning, forgetting, data curve
uv run lab lora-params           # LoRA size and weights memory
```

## The three most common mistakes

1. **Trusting a random split.** Duplicates and the absence of drift inflate the score; split by time
   and deduplicate first.
2. **Letting a leak in through the context.** A field written after the fact can enter through the
   prompt, the example pool or the retrieval index, not only through features.
3. **Fine-tuning without an old-data gate.** Score new *and* old data; low rank does not prevent
   forgetting, replay and early stopping do help.
