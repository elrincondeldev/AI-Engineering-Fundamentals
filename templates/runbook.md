🌐 English · [Español](runbook.es.md)

# Runbook: <service name>

> A runbook tells the person on call what to do, step by step, when something goes wrong. Write it
> for someone tired at 3 a.m. who has never seen the service: short sentences, exact commands, links.

| Field | Value |
|---|---|
| Service | |
| Owner team and on-call rota | |
| Dashboards | |
| Logs and traces | |
| Dependencies | *model provider, vector index, tools, databases* |
| Last reviewed | `<YYYY-MM-DD>` |

## Service overview

*What it does, who depends on it, and the normal ranges of the key metrics (requests per minute,
p95 latency, error rate, cost per hour, abstention rate).*

## Alerts and first actions

| Alert | Likely causes | First actions | Escalate to |
|---|---|---|---|
| Error rate above threshold | Provider outage, expired key, schema change | Check provider status; switch to the fallback provider | |
| p95 latency above threshold | Provider slowness, long contexts, retries | Check token counts and retry rate; lower the context budget | |
| Cost per hour above budget | Prompt growth, retry storm, abuse | Check top users and prompt sizes; enable rate limits | |
| Spike in "I don't know" answers | Index not updated, retrieval broken | Run the retrieval smoke test; check last index build | |
| Injection or data-leak detector fires | Malicious content in sources or inputs | Block the source; preserve traces; page Security | |

## Common procedures

### Switch to the fallback model provider

1. *Exact command or configuration change*
2. *How to verify it worked*

### Roll back a prompt or model version

1. *Exact steps*
2. *How to verify*

### Rebuild the retrieval index

1. *Exact steps*
2. *How to verify (smoke-test questions and expected sources)*

## Verification after any change

- [ ] Smoke tests pass
- [ ] Evaluation suite meets the launch bar
- [ ] Dashboards back to normal ranges

## Contacts

| Role | Name | How to reach |
|---|---|---|
| | | |

---

[🏠 Course home](../README.md) · [Postmortem template →](postmortem.md)
