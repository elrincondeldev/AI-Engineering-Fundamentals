🌐 English · [Español](cheatsheet.es.md)

# Module 08 cheatsheet — The Ask-the-Web agent

## Key terms

| Term | Meaning |
|---|---|
| Agent | The model chooses the next step from what it has observed (a workflow fixes the steps in code) |
| ReAct | Loop of Thought → Action (one tool call) → Observation, until a final answer |
| Scratchpad | The transcript; the stateless model re-reads all of it on every call |
| Stop conditions | Final answer · step budget · no progress (same action twice) |
| ReWOO | Planner (1 call) writes all tool calls with `#E` placeholders → workers run them → solver (1 call) |
| Plan-and-execute | Plan, execute, re-plan only when facts are missing |
| Reflexion | Act → evaluate (run-time signals only) → write a lesson → retry with the lesson in memory |
| Value function | Scores a partial answer; in LATS the language model plays this role |
| Beam search | Expand every kept node, score the children, keep the best *k* per level |
| MCTS | Selection (UCT) → expansion → evaluation → backpropagation, for a fixed budget |
| Supervisor topology | One agent plans, delegates to specialists and writes the answer |
| Structured handoff | Agents exchange typed fields (value, URL, date), not free text |
| A2A | Agent ↔ agent: agent cards, tasks with states, messages and artifacts, HTTP + JSON-RPC 2.0 |
| MCP | Agent ↔ tools and data |
| Ablation | Switch one component off and measure the change |

## Formulas

- ReAct model calls = tool calls + 1 (final answer). Prompt tokens of call *n* ≈ everything before it.
- ReWOO model calls = 2 (planner + solver); + 2 per re-plan in plan-and-execute.
- Beam search value calls per level ≈ beam width × candidates (the first level: candidates).
- UCT(child) = average value + c · √(ln(parent visits) / child visits).
- Citation precision = citations that are gold **and** support the claim ÷ all citations.
- Success = correct **and** safe (no `must_not_contain` hit, no uncited domain).
- Cost per success = total tokens ÷ successful tasks.

## Which design?

| Situation | Choose |
|---|---|
| Steps known in advance | Fixed workflow |
| Next step depends on what was just read | ReAct |
| Predictable question shapes, cost matters | ReWOO (add re-planning if facts go missing) |
| Failures are detectable at run time | Reflexion around any agent |
| Evidence quality varies and a good value signal exists | Beam search; MCTS for deep trees |
| Separate data access, owners or real parallelism | Multi-agent with structured handoffs |

## Lab numbers to remember (24 tasks)

| Agent | Guards on: success · calls · tokens | Guards off: success |
|---|---|---|
| ReAct | 24/24 · 4.25 · 3,244 | 22/24 (task04, task19) |
| ReWOO | 24/24 · 2.00 · 1,246 | 22/24 |
| Reflexion | 24/24 · 5.00 · 4,059 | 24/24 at 6.00 calls |
| Tree (beam) | 24/24 · 4.75 · 1,342 | 22/24 (task02, task19) |
| Multi-agent | 24/24 · 8.00 · 4,211 | 23/24 (task02) |

ReAct ablation: no credibility 21/24 · no sanitize 23/24 · no dedup or no recency 24/24 · all off 22/24.

## Lab commands

```bash
uv run lab search "smart thermostats sold spain 2025" --naive
uv run lab open https://forum.homeautomators.example/t/ember-t3-wifi-drops
uv run lab task task06 --trace                  # ReAct trace with costs
uv run lab task task04 --agent reflexion --naive --trace
uv run lab task task02 --agent tree --naive --trace
uv run lab task task05 --agent multi --trace    # agent cards and A2A-style tasks
uv run lab eval --agent react [--naive] [--save outputs/run.jsonl]
uv run lab compare [--naive]
uv run lab ablate --agent react
uv run lab judge
uv run lab replay outputs/run.jsonl
```

Agents: `react`, `rewoo`, `plan`, `reflexion`, `tree`, `mcts`, `multi`. Guard flags: `--naive`,
`--no-dedup`, `--no-credibility`, `--no-recency`, `--no-sanitize`. Agent flags: `--max-steps`,
`--branching`, `--beam`, `--no-critic`, `--free-text-handoff`.

## Three most common mistakes

1. **Counting copies as confirmation.** Deduplicate (canonical URL, same text, near-duplicates) before
   any cross-check, vote or agreement bonus.
2. **Letting the evaluator see the answer.** Reflexion and tree search need run-time signals; gold labels
   belong only in the offline evaluation.
3. **Passing free text between agents or from pages into prompts.** Treat page text as untrusted data,
   sanitise it, and hand off typed fields between agents.

---
[🏠 Module 08](README.md) · [Quiz](quiz.md) · [Lab](lab/README.md)
