🌐 English · [Español](quiz.es.md)

# Module 08 quiz — The Ask-the-Web agent

13 questions, about 20 minutes. Answer first, then open the explanation. Questions marked 🔁 review
earlier modules.

### Q1 — Choosing autonomy

Hearthwise wants a daily job that downloads the returns-policy page and posts a three-line summary to a
channel. Which design fits best?

- A. A ReAct agent with `search_web` and `open_page`
- B. A fixed workflow: fetch the page, summarise, post
- C. A supervisor with a researcher and a fact-checker
- D. Monte Carlo tree search over possible summaries

<details><summary>Answer</summary>

**B.** The steps are known in advance and nothing depends on intermediate results. Every extra level of
autonomy (A, C, D) adds cost, failure modes and evaluation work without adding capability.

</details>

### Q2 — Spot the bug

```python
for _ in range(max_steps):
    text = provider.complete(messages).text
    kind, arg = parse_action(text)
    if kind == "final":
        return text
    obs = tools.call(kind, arg)
    messages.append(Message("assistant", text + "\nObservation: " + obs))
```

What is wrong, and what can a real model do because of it?

<details><summary>Answer</summary>

Two problems. The observation is written inside the *assistant* message, so the model cannot tell what
it said from what the tool returned; it should be a separate message from the harness. And there is no
`stop=["\nObservation:"]`, so a real model can write its own fake observation and continue from it.
(Also, nothing happens when the budget runs out: the loop just ends without an answer.)

</details>

### Q3 — Scratchpad cost (calculation)

A ReAct run makes four model calls with prompt sizes of 300, 600, 900 and 1,200 tokens. How many prompt
tokens did it use, and how many would a fifth step at the same growth rate add?

<details><summary>Answer</summary>

300 + 600 + 900 + 1,200 = **3,000** prompt tokens. A fifth call would carry about 1,500 tokens, so the
total would rise to about 4,500: each step adds more than the previous one because the whole transcript
is re-sent.

</details>

### Q4 — ReWOO

On the lab's 24 tasks with every guard on, ReWOO matches ReAct's success with 2.00 model calls per task
instead of 4.25. Which statement explains where ReWOO *loses*?

- A. It needs more model calls for multi-hop questions
- B. It cannot adapt its plan to what the observations contain
- C. It cannot cite sources
- D. It always answers "unknown" on unanswerable tasks

<details><summary>Answer</summary>

**B.** The planner writes every tool call before seeing any evidence, so it reads two results even when
the first is decisive (3.88 tool calls per task against 3.25) and cannot reformulate a query that
returned nothing useful. A is false: it always uses two calls. C and D are not specific to ReWOO.

</details>

### Q5 — Reflexion's evaluator

A teammate proposes this Reflexion evaluator: "retry if the answer does not match the reference answer
in tasks.jsonl". What do you tell them?

<details><summary>Answer</summary>

That it leaks the gold answer into the agent. The evaluation would look excellent and say nothing about
production, where no reference exists. The evaluator must use run-time signals only: spam signals in a
cited page, injected text echoed in the answer, a newer source that disagrees, an "unknown" answer.

</details>

### Q6 — Tree search arithmetic (calculation)

A beam search has 2 levels, 3 candidate pages per level and a beam width of 2. How many value-model
calls does it make, assuming every candidate yields a node?

<details><summary>Answer</summary>

Level 1: 3 children of the root. Level 2: 2 kept nodes × 3 candidates = 6. Total **9** value calls. The
cost grows with branching factor times beam width per level, which is why the lab groups facts that
share a page into one level.

</details>

### Q7 — A fooled value function

With every guard off, the lab's beam search answers 200 for "How many devices can a single HomeMesh 2.0
network support?" (correct: 250). What is the best fix?

- A. Widen the search to 5 candidates
- B. Remove duplicate pages before computing agreement, or show credibility tiers to the value model
- C. Switch to Monte Carlo tree search (MCTS)
- D. Lower the temperature

<details><summary>Answer</summary>

**B.** The forum post and its `?utm_source=` copy "agree" on 200 and get the agreement bonus. A makes it
worse (21/24 instead of 22/24 in the lab), C uses the same value function, and D changes nothing in a
deterministic value model.

</details>

### Q8 — Multi-agent blast radius

The researcher agent reads a page with an injected instruction and repeats it in its free-text summary.
Which design change most directly stops the instruction from reaching the final answer?

<details><summary>Answer</summary>

Make the handoff structured: the supervisor accepts only typed fields (value, web address, date) from the
researcher, never its prose. In the lab, the free-text handoff let the advert reach the user; the
structured handoff blocked it even with the sanitiser off. Least privilege for the agent that reads the
web limits the damage further.

</details>

### Q9 — A2A or MCP?

Match each need to the Agent2Agent (A2A) protocol or the Model Context Protocol (MCP): (1) expose `search_web` to any agent framework; (2) let a partner's
pricing agent, whose code you cannot see, take a "compare prices" task and report back later.

<details><summary>Answer</summary>

(1) **MCP**: it connects agents to tools and data. (2) **A2A**: the other side is an opaque agent that
works on a task with its own lifecycle (submitted, working, input-required, completed...) and returns
artifacts; you discover it through its agent card.

</details>

### Q10 — Reading an ablation

With all guards on, ReAct succeeds on 24/24. Switching off only credibility gives 21/24; switching off
every guard gives 22/24. How can removing *more* guards fail *fewer* tasks?

<details><summary>Answer</summary>

Guards interact. With credibility off but recency on, "prefer the newest source" picks a newer forum post
and a newer spam page over the specification and the MarketLens report. With recency also off, conflicts
fall back to counting agreeing sources and then to "first seen", which happens to favour the better
pages on two of those tasks (task02 and task21). Ablate guards
one at a time *and* in combination.

</details>

### Q11 🔁 (module 06)

In module 06 the support chatbot had to refuse when the help centre did not contain the answer. Which
two metrics in this module play the same role for the Ask-the-Web agent?

<details><summary>Answer</summary>

"Unknown" accuracy on the unanswerable tasks (did it abstain when it should?) and the false-"unknown"
count on answerable tasks (did it abstain when it should not?). Reporting only the first rewards an agent
that answers "unknown" to everything.

</details>

### Q12 🔁 (module 05)

The agent's search uses BM25 (Best Matching 25). The query "smart thermostats sold spain 2025" ranks a keyword-stuffed spam
page first when re-ranking is off. Why does BM25 favour it, and which module 05 idea limits the effect?

<details><summary>Answer</summary>

BM25 rewards query terms that appear often, and keyword stuffing repeats them. Term-frequency saturation
(the `k1` parameter) limits how much each repetition adds, but it cannot tell an honest page from a
stuffed one. That is why the lab adds credibility re-ranking and spam signals on top of retrieval.

</details>

### Q13 🔁 (module 07)

In module 07 you exposed tools through MCP and reviewed their permissions with Noor. The Ask-the-Web
agent has only `search_web` and `open_page`. Why does that matter for indirect prompt injection?

<details><summary>Answer</summary>

Least privilege limits the blast radius. An injected instruction can at most change what the agent
*says*; it cannot send e-mail, buy anything or change data, because no such tool exists. That is also
why the remaining risk (a polluted answer) is tested with `must_not_contain` and uncited-domain checks.

</details>

## 📊 Scoring guide

| Correct | What it means |
|---|---|
| 12–13 | Ready for module 09. |
| 9–11 | Solid; re-read the self-checks of the lessons where you missed questions. |
| 6–8 | Re-run the lab steps for those lessons and redo their "Try it" sections. |
| 0–5 | Work through the lessons again with the lab open beside them. |

## ⏭️ Next

[Module 09 — Reasoning models →](../09-reasoning-models/README.md)
