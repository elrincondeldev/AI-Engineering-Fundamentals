🌐 English · [Español](cheatsheet.es.md)

# Module 03 cheatsheet — Build an LLM Playground

## Components: one job each

| Component | Job | Lab file |
|---|---|---|
| Client | Input and output | `cli.py` |
| API layer | Auth, rate limits, quotas | not built |
| Orchestrator | Runs the stages in order | `playground.py` |
| Input guardrails | Redact PII, flag injection, data boundary | `redact.py`, `guardrails.py` |
| Prompt assembly | System + history + message within budget | `conversation.py` |
| Model gateway | One interface; timeouts, retries, fallback | `resilience.py`, `providers.py` |
| Output guardrails | `empty`, `truncated`, `pii_in_output`, banned claims | `guardrails.py` |
| Conversation store | History per session (redacted) | `Conversation` |
| Tracing | One JSON line per request, no raw text | `tracing.py` |
| Evaluation loop | Traces → test cases → gate changes | `compare.py` |

## Formulas

| What | Formula |
|---|---|
| Time to first token (TTFT) | overhead + prefill per token × prompt tokens |
| Total latency | TTFT + decode per token × (completion tokens − 1) |
| Decode speed | (n − 1) ÷ (total − TTFT) |
| End-to-end throughput | n ÷ total |
| Cost per request | (prompt × input price + completion × output price) ÷ 1,000,000 |
| Requests per month | users × requests per user per day × working days |
| Input budget | context window − `max_tokens` |
| Prompt at turn n | ≈ S + (n − 1) × k + new message |
| Total over N turns | ≈ N × (S + message) + k × N × (N − 1) ÷ 2 (quadratic) |
| Success with r retries | 1 − p^(r + 1) for independent failures with probability p |
| Backoff ceiling | min(cap, base × 2^(retry − 1)); full jitter waits uniform(0, ceiling) |

Simulated profiles (invented): `fake-small` 120 ms overhead, 0.25 ms/prompt token, 20 ms/token,
window 1,024; `fake-large` 300 ms, 0.60 ms, 45 ms, window 8,192. Example prices per million tokens:
€0.10/€0.40 and €1.00/€4.00.

## Error kinds

| Kind | Retry | Fall back |
|---|---|---|
| `timeout`, `rate_limit` (429), `server` (5xx), `network` | yes, backoff + jitter | yes |
| `auth` (401/403) | no | yes |
| `bad_request` (400), `context_length` | no: fix the request | no |

Size timeouts so that every attempt plus its wait fits the deadline, with room for the fallback.

## Truncation policies (20-turn recording, budget 768)

| Policy | Total sent | Over budget | No system prompt | Remembers the order |
|---|---|---|---|---|
| `keep_all` | 12,657 | 8 turns | 0 | yes |
| `sliding_window` (6 messages) | 3,110 | 0 | 17 turns | no |
| `pin_system` | 11,046 | 0 | 0 | no |
| `summarise` | 11,128 | 0 | 0 | yes |

## Redactor levels (1,224 tickets: 1,224 emails, 82 phones, 82 names)

| Level | Names caught | False positives | Precision | Leaking tickets |
|---|---|---|---|---|
| `naive` | 0/82 | 200 (dates) | 0.867 | 82 |
| `regex` | 0/82 | 0 | 1.000 | 82 |
| `names` | 75/82 | 0 | 1.000 | 7 (capital letters) |

## Lab commands

```bash
uv run lab ask "How long do I have to return a Glow A19?" [--stream] [--flaky server,rate_limit]
uv run lab chat                 # /system /temp /model /max /reset /history /cost /quit
uv run lab retry                # scripted failures + 1,000-request reliability table
uv run lab budget [--budget 400] [--show-summary]
uv run lab redact [--ticket T00008]
uv run lab guard                # injection tripwires + data boundary
uv run lab compare [--show]     # models x temperatures x 10 seeds
uv run lab cost [--users 60 --requests-per-user 25 --days 21]
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab ask "..."   # local model (untested here)
```

## Three most common mistakes

1. **Guardrails in the client.** Enforce redaction and the data boundary in the orchestrator; fail
   closed for unknown providers.
2. **Retrying everything, without jitter or a deadline.** Classify first; fail fast on bad requests;
   add a circuit breaker for outages.
3. **A sliding window that drops the system prompt, and a budget that forgets `max_tokens`.** Pin the
   system prompt, budget by tokens, keep user/assistant pairs together.

---

[🏠 Module 03](README.md) · [Quiz](quiz.md) · [Lab](lab/README.md)
