🌐 English · [Español](cheatsheet.es.md)

# Module 10 cheatsheet — Deep research and local deployment

## The research loop

| Stage | Contract | Best done by |
|---|---|---|
| Plan | Question → 2–6 sub-queries with aspect and entities (validated JSON, fallback = the question) | Reasoning model |
| Search | Sub-query → top-k pages (BM25 in the lab, k = 4) | Search engine |
| Filter | Page → credibility 0–1 with reasons; skip below 0.4 | Code |
| Read | Sanitised page → verbatim quotes; keep only quotes found in the page | Small model + code |
| Notes | Passage → note with URL, publisher, date, credibility, recency, relevance per sub-query | Code |
| Reflect | Notes per sub-query → missing entities → follow-up searches | Code or model |
| Write | Numbered notes + gaps → report with `[n]` citations, gaps and caveats | Reasoning model |
| Grade | Report → facts covered and supported, valid citations, domains, low-credibility, traps, PII | Code |

**Stop when:** all sub-queries covered · no new evidence in an iteration · budget spent (iterations, searches, pages, tokens).

## Five failure modes

| Failure | Defence |
|---|---|
| Hallucinated citation | Drop markers with no source; check every cited number against the cited page |
| Low-credibility source | Source policy + content signals (stuffing, shouting, sales language, forum posts) |
| Injected instructions | Sanitise before any model reads; verify quotes after; minimal agent tools |
| Outdated facts | Recency weight (half-life 365 days); newest page of the same publisher wins |
| Premature stopping | Coverage-based stopping; report distinct domains next to facts |

## Key formulas

- **Note score** = relevance × credibility × recency; recency = 0.5^(age in days / half-life)
- **Bits per weight** = bits + 16 / group size (4-bit, group 32 → 4.5)
- **Weights (bytes)** = parameters × bits per weight / 8
- **KV cache (bytes)** = 2 × layers × KV heads × head dim × bytes per value × tokens × batch
- **Total memory** = (weights + KV cache) × (1 + overhead); plan with 75% of RAM
- **Decode ceiling (one user)** = memory bandwidth / weight bytes
- **Symmetric quantisation:** scale = max|w| / (2^(bits−1) − 1); code = clip(round(w / scale))
- **End-to-end latency** = TTFT + TPOT × (output tokens − 1)
- **Speculative decoding:** tokens per pass = (1 − α^(k+1)) / (1 − α); speed-up = tokens / (k·c + 1)
- **API cost/day** = in tokens/1M × price in + out tokens/1M × price out
- **Local cost/day** = hardware / (months × 30) + kW × busy hours × price per kWh
- **Servers** = ceil(peak tokens/s × (1 + headroom) / tokens/s per server)

## Numbers from the lab (by our formula or simulator)

| Item | Value |
|---|---|
| 8B params, 4-bit g32, 8,192 tokens (32 layers, 8 KV heads, dim 128) | 4.50 + 1.07 + 0.56 = 6.13 GB |
| Same at 16-bit | 18.78 GB (does not fit 12 GB) |
| KV cache per token (that shape) | 128 KiB |
| int4 per-tensor vs group-32 output error | 0.92 vs 0.11 |
| TTFT p50: static vs continuous batching (batch 16, simulated) | 10,453 ms vs 98 ms |
| KV waste: contiguous vs paged (16-token pages) | 89.5% vs 0.9% |
| Ablation that leaks injection and PII | no-credibility + no-sanitiser: 2 traps, 2 PII |

## Decision table: which model on a laptop?

| Question | If yes | If no |
|---|---|---|
| Does weights + KV + overhead fit in 75% of RAM? | Next question | Fewer bits, smaller model or shorter context |
| Is the decode ceiling fast enough for the task? | Next question | Fewer bits or smaller model |
| Does it pass your task grades (`uv run lab grade`)? | Ship the smallest passing option | Larger model or more bits |

## Serving vocabulary

| Term | Meaning |
|---|---|
| TTFT | Time to first token; includes queueing |
| TPOT | Time per output token after the first |
| Continuous batching | Requests join and leave the batch at every decode step |
| PagedAttention | KV cache stored in small fixed-size blocks mapped by a block table |
| Speculative decoding | Draft model proposes k tokens; target verifies them in one pass |

## Self-hosted endpoint checklist

- Bind to localhost or sit behind a gateway; authenticate every request; per-team keys and rate limits.
- TLS for anything off the machine; pin model files by checksum; minimal agent tools.
- Log metadata, not prompts; watch TTFT/TPOT p95, queue depth, KV-cache use, tokens/s, errors.

## Lab commands

```bash
uv run lab research rq1 --trace        # one report + trace -> outputs/rq1.md
uv run lab grade                        # all four questions
uv run lab grade-file rq1 outputs/rq1-edited.md
uv run lab ablate                       # defences off one at a time
uv run lab search "query" -k 5          # BM25 + credibility reasons
uv run lab memory --ram 16 --bandwidth 100
uv run lab memory --sweep --ram 16
uv run lab quantize
uv run lab serve-sim && uv run lab kv-cache && uv run lab speculate
uv run lab cost --peak-tokens-s 400
```

## Three most common mistakes

1. **Trusting a clean-looking report.** Grade it: covered is not supported, and two sources are not four.
2. **Sizing a model from its file size.** Add the KV cache (context × batch) and overhead, and leave headroom.
3. **Treating "internal" or "local" as security.** Local is a configuration; the endpoint still needs authentication, and searches still leave the machine.
