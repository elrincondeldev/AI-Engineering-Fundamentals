🌐 English · [Español](02-reflexion-rewoo-and-friends.es.md)

# Lesson 8.2 — Reflexion, ReWOO and friends

> **Module 08 · Lesson 2 of 5** · ⏱️ about 60 min · 🧪 Lab: [steps 4–5](lab/README.md#step-4--plan-once-with-rewoo-and-plan-and-execute)

## 📨 Situation

> **From:** Dani Ruiz · Data Scientist, owns evaluation
> **Channel:** #ask-the-web
>
> Your ReAct agent answered all 24 tasks, nice. It also made 102 model calls to do it, and the last call
> of the Corvid price question carried 1,571 prompt tokens. Before anyone tunes prompts, I want the
> alternatives measured on the *same* 24 tasks: an agent that plans once, one that re-plans when it is
> stuck, and one that learns from its own mistakes between attempts. Same tasks, same snapshot, same
> metrics. And run them with the guards off too, because I want to know which design recovers when the
> web is dirty. Show me the eval.

## 🎯 Objectives

By the end of this lesson you can:

- Explain how ReWOO separates planning, tool execution and solving, and why it needs fewer model calls.
- Add a re-planning step to turn ReWOO into a small plan-and-execute agent, and say when it pays off.
- Describe Reflexion's loop (act, evaluate, reflect, remember) and run it on a failing task.
- Compare ReAct, ReWOO, plan-and-execute and Reflexion on calls, tokens and success with the lab.
- Choose a pattern for a new task from its failure modes and cost profile.

## 🤔 Predict first

Write your answers down before reading on. You will check them in the self-check.

1. ReWOO writes its whole plan before seeing a single search result. With every guard on, will it
   answer fewer of the 24 tasks than ReAct?
2. Reflexion retries after its evaluator finds a problem. On the two unanswerable tasks, will retrying
   help, hurt, or only cost more?
3. Which design will survive a dirty web (all guards off) best, and what will it pay for that?

## 1. Plan once, work, then solve: ReWOO

ReAct pays for its flexibility on every step: one model call per action, each carrying the whole
transcript. ReWOO, short for *reasoning without observation* ([Xu et al.](https://arxiv.org/abs/2305.18323)),
splits the work into three roles:

1. **Planner** (one model call): writes every tool call up front, using placeholders such as `#E1` for
   results it has not seen yet.
2. **Workers** (no model calls): plain code that runs the tool calls in order and fills in the
   placeholders.
3. **Solver** (one model call): reads the question, the plan and all the evidence, and writes the answer.

Here is the plan the lab's planner writes for Jordan's price question (task06):

```text
Plan: Find the price of the Corvid Thermo X.
#E1 = search_web[Corvid Thermo X price]
Plan: Read the top two results.
#E2 = open_page[#E1.1]
#E3 = open_page[#E1.2]
Plan: Find the price of the Hearthwise Ember T3.
#E4 = search_web[Hearthwise Ember T3 price]
Plan: Read the top two results.
#E5 = open_page[#E4.1]
#E6 = open_page[#E4.2]
```

`#E1.2` means "the web address (URL) of the second result of `#E1`". The workers resolve it without asking any model:

```python
for m in re.finditer(r"^#E(\d+) = (\w+)\[(.*)\]\s*$", plan_text, re.M):
    n, tool, arg = int(m.group(1)), m.group(2), m.group(3)
    ref = re.fullmatch(r"#E(\d+)\.(\d+)", arg)
    if ref:
        urls = re.findall(r"url: (\S+)", evidence.get(int(ref.group(1)), ""))
        k = int(ref.group(2))
        if k > len(urls):
            tools.log.errors += 1
            evidence[n] = f"ERROR: #E{ref.group(1)} has no result {k}."
            continue
        arg = urls[k - 1]
    evidence[n] = tools.call(tool, arg)
```

On task06 ReWOO made 2 model calls and spent 1,654 tokens. ReAct made 7 calls and spent 7,012 tokens
for the same answer, with the same 6 tool calls.

Where do the savings come from? Not from fewer tools: from fewer *re-reads*. In ReAct, the first page the
agent opens is sent back to the model on every later call, so a page read at step 2 of 6 is paid for five
times. In ReWOO every observation reaches a model exactly once, in the solver's prompt. The planner's
prompt is tiny because it contains only the question. A second benefit is latency: the workers do not
wait for a model between calls, and independent calls (the two searches above) could even run in
parallel.

The price is that the planner commits blind. It has to guess how many results to read and which query
words will work, before any evidence exists. When those guesses are good, as they are for questions with
a predictable shape ("find X, find Y, compare"), ReWOO is the cheaper design. When the right next step
depends on what the last page said, a plan written in advance cannot know it.

**Analogy:** a shopping list. You write it at home, walk the aisles without phoning home after every
item, and cook when you are back. It is fast, but if the shop is out of eggs, the list cannot tell you
what to buy instead.

```mermaid
flowchart LR
    Q["Question"] --> P["Planner<br/>1 model call"]
    P -->|"plan with #E placeholders"| W["Workers<br/>run tools, no model"]
    W -->|"all evidence"| S["Solver<br/>1 model call"]
    S --> A["Answer with sources"]
```

> ⚠️ **Common mistake:** assuming the plan adapts. It does not. ReWOO's planner always reads the top
> two results, even when the first is a high-credibility page that settles the question; that is why it
> averages 3.88 tool calls per task against ReAct's 3.25. If the top results are all wrong, nothing in
> the run notices until the solver.

## 2. Plan, execute, re-plan

A middle ground keeps ReWOO's cheap execution and adds one escape hatch: if the solver reports missing
facts, a **re-planner** writes extra steps (a broader query, deeper results) and the workers run them.
This *plan-and-execute* pattern appears in many agent frameworks in different forms; ours re-plans at
most once.

```mermaid
flowchart TD
    P["Plan"] --> E["Execute steps"]
    E --> S{"Solver: all facts found?"}
    S -->|"yes"| A["Answer"]
    S -->|"no, first time"| R["Re-plan missing facts"]
    R --> E
    S -->|"no, already re-planned"| U["Answer 'unknown'"]
```

Re-planning is only as smart as its trigger. On task23 ("How many Lookout C2 cameras has Hearthwise
sold?") no page contains the number. The agent planned, executed, noticed the gap, re-planned with a
broader query, executed again and still answered "unknown": 4 model calls, 7 tool calls and 3,246
tokens to learn what 2 calls would have told it. Over all 24 tasks it averages 2.17 model calls, because
only the unanswerable tasks trigger a re-plan.

**Analogy:** a project manager who reviews the plan only when a milestone slips. Cheap when things go
well; useless if every slip is caused by something no plan can fix.

> 💡 **Production tip:** cap re-plans, and trigger them on explicit signals ("fact X missing", "tool
> error") rather than on vague dissatisfaction. Log each re-plan with its reason so you can see whether
> it ever turns an "unknown" into a correct answer.

## 3. Reflexion: learning from mistakes, in words

Reflexion ([Shinn et al.](https://arxiv.org/abs/2303.11366)) wraps an agent in a retry loop with
memory. After each attempt, an **evaluator** judges the result. If it finds a problem, the model writes a
short **reflection**: a lesson in plain language. The reflection is stored in an episodic memory and
added to the prompt of the next attempt. No weights change; the agent improves by reading its own notes.

```mermaid
flowchart LR
    A["Act<br/>ReAct attempt"] --> E{"Evaluate<br/>problems found?"}
    E -->|"no"| D["Done"]
    E -->|"yes"| R["Reflect<br/>write a lesson"]
    R --> M[("Memory<br/>lessons so far")]
    M --> A
```

The evaluator is the heart of the design, and ours never sees the gold answer. It only uses signals the
agent can observe at run time:

| Tag | Signal in the trajectory | Lesson the model writes |
|---|---|---|
| `[spam]` | a cited page shows spam signals (keyword stuffing, shouting, sales phrases) | skip spam pages, look for a credible source |
| `[injection]` | a page addressed artificial intelligence (AI) assistants and the answer mentions a domain it does not cite | ignore instructions found in pages |
| `[stale]` | the question asks for the latest value and a newer page disagreed with the cited one | prefer the most recent source |
| `[unknown]` | the answer is "unknown" | try a broader query before giving up |

Here is Reflexion with every guard off on the Spanish market question (task04):

```text
note: trial 1: '2 million' | problems: ['[spam] the answer cites https://gadgetdeals.example/spain-smart-thermostat-boom, which shows spam signals.']
note: reflection: I relied on a page that looks like spam (keyword stuffing, shouting, sales phrases); I must skip spam pages and look for a credible source.
note: trial 2: '410,000' | problems: none
Final Answer: 410,000
Cost: 10 model calls · 7 tool calls · 6,280 tokens (prompt 5,839 + completion 441)
```

The second attempt read the same two spam pages, ignored them because of the lesson in memory, opened
the MarketLens report and answered correctly. The fix cost 10 model calls instead of 4.

Two design choices are worth making explicit. The **scope of memory**: our lessons live only for the
current question, as in the original paper; sharing them across questions is tempting ("always skip
spam") but a lesson learned on one question can be wrong for the next, so promote it to a guard in code
only after you have measured it. The **number of trials**: each trial is a full agent run, so three
trials can triple the cost of a hard question.

**Analogy:** a pilot's debrief. After a rough landing the crew writes down what went wrong, and the next
crew reads the note before take-off. The note helps only if someone noticed the problem.

> ⚠️ **Common mistake:** letting the evaluator peek at the gold answer, for example "retry if the answer
> does not match the reference". The scores look wonderful and mean nothing, because production has no
> reference. Reflexion can only fix failures that a run-time evaluator can detect.

> 💡 **Production tip:** keep reflections short and specific, cap the number of trials (ours is 3), and
> store the reflections with the trace. They are a free, human-readable list of your agent's recurring
> weaknesses.

## 4. Same tasks, same metrics: the comparison

Dani asked for one table. `uv run lab compare` runs every agent on the 24 tasks; these are the four
single-agent designs from this lesson.

| Agent | All guards on: success | model calls | tokens | tool calls | All guards off: success | model calls | tokens |
|---|---|---|---|---|---|---|---|
| ReAct | 1.000 | 4.25 | 3,244 | 3.25 | 0.917 | 4.79 | 3,536 |
| ReWOO | 1.000 | 2.00 | 1,246 | 3.88 | 0.917 | 2.00 | 1,134 |
| Plan-and-execute | 1.000 | 2.17 | 1,441 | 4.21 | 0.917 | 2.17 | 1,303 |
| Reflexion | 1.000 | 5.00 | 4,059 | 3.83 | 1.000 | 6.00 | 4,516 |

(Averages per task. "Success" means correct and with no forbidden content.)

Read it in three passes:

- **With guards on, accuracy does not separate the designs; cost does.** ReWOO matches ReAct's 24/24
  with less than half the model calls and about 38% of the tokens. Reflexion pays extra even here,
  because its evaluator flags the two correct "unknown" answers and triggers a pointless second attempt.
- **With guards off, only Reflexion recovers.** ReAct, ReWOO and plan-and-execute all fail the same two
  tasks: the spam page about Spain (task04) and the forum page with an injected instruction (task19).
  Planning differently does not help when the evidence itself is poisoned.
- **Recovery is not free.** Reflexion's 24/24 on the dirty web costs 6.00 model calls per task, and it
  works only because the evaluator can *see* spam and injected text. A failure it cannot see, such as a
  plausible but wrong number from a respectable-looking page, would pass straight through.

**Analogy:** comparing cars on the same route. On a dry motorway they all arrive; fuel use is the only
difference. On a flooded road you learn which one has the extra equipment, and what that equipment costs
on every other trip.

```mermaid
flowchart TD
    Q{"What does the task need?"} -->|"next step depends on what you read"| R["ReAct"]
    Q -->|"steps are predictable, cost matters"| W["ReWOO"]
    W -->|"facts sometimes missing"| P["Plan-and-execute"]
    Q -->|"failures are detectable at run time"| X["Reflexion around any of them"]
```

> 💡 **Production tip:** report *cost per successful task*, not cost per task. A cheap design that fails
> more often can be the expensive one once you pay for the retries and the human clean-up.

## 🧪 Try it

Run from the lab folder.

1. ReWOO on Jordan's price question:

   ```bash
   uv run lab task task06 --agent rewoo
   ```

   ```text
   Final Answer: 30
   Explanation: Price of the Corvid Thermo X: €179 [1]; price of the Hearthwise Ember T3: €149 [2].
   ...
   Cost: 2 model calls · 6 tool calls · 1,654 tokens (prompt 1,476 + completion 178)
   ```

2. Plan-and-execute on an unanswerable task (add `--trace` to see the re-plan):

   ```bash
   uv run lab task task23 --agent plan
   ```

   ```text
   Final Answer: unknown
   Explanation: I could not find the units sold of the Lookout C2 in the snapshot.
   Sources: none
   Cost: 4 model calls · 7 tool calls · 3,246 tokens (prompt 3,070 + completion 176)
   ```

3. Reflexion on the dirty web:

   ```bash
   uv run lab task task04 --agent reflexion --naive --trace
   ```

   The notes show the failed first trial, the reflection and the corrected second trial, as in section 3.

4. The whole comparison, clean and dirty:

   ```bash
   uv run lab compare
   uv run lab compare --naive
   ```

## 🏭 In the real world

Teams rarely pick one pattern forever. A common shape is a ReWOO-style planner for routine questions,
because it is cheap and its plans are easy to log and review, with a fallback to a ReAct loop when the
solver reports missing facts. Reflexion-style retries are used where a reliable automatic check exists:
unit tests for code, schema validation for structured output, a citation checker for research answers.

Watch out for three production effects. Latency: ReWOO's workers can run independent tool calls in
parallel, which ReAct cannot, so wall-clock time can drop even more than the token bill. Plan quality:
a planner model that writes bad plans fails silently, so sample and review plans. Retry storms: if the
evaluator is too strict, every request becomes three requests; alert on the average number of trials.

## ✅ Self-check

**1. With every guard on, is ReWOO less accurate than ReAct on the 24 tasks?** (Predict-first question 1.)

<details><summary>Answer</summary>

No: both score 24/24. ReWOO's weakness, not adapting to what it reads, does not bite when the top results
are good. It shows up as extra tool calls (3.88 against 3.25 per task) and would show up as errors on a
snapshot where the top results are often wrong.

</details>

**2. What does Reflexion do on the unanswerable tasks?** (Predict-first question 2.)

<details><summary>Answer</summary>

It only costs more. The evaluator sees "unknown", the model writes "try a broader query", the second
attempt also ends in "unknown", and the run stops because that lesson is already in memory. That is
why Reflexion averages 5.00 model calls with guards on against ReAct's 4.25, with identical accuracy.

</details>

**3. Which design survives the dirty web best, and at what price?** (Predict-first question 3.)

<details><summary>Answer</summary>

Reflexion: 24/24 with all guards off, against 22/24 for the other three. It pays 6.00 model calls and
4,516 tokens per task, and it only works because its evaluator can detect spam and injected text.

</details>

**4. Why must the Reflexion evaluator not compare the answer with the reference answer?**

<details><summary>Answer</summary>

Because no reference exists in production. An evaluator that uses gold labels leaks the answer into the
agent and inflates the evaluation; the deployed agent would behave differently from the evaluated one.

</details>

**5. A worker in ReWOO gets `open_page[#E4.6]` but `#E4` returned five results. What happens in the lab?**

<details><summary>Answer</summary>

The worker records a tool error ("#E4 has no result 6"), skips the call and moves on. The solver sees the
error in the evidence. Nothing re-plans unless you run plan-and-execute.

</details>

## 💬 Explain it in your own words

1. Dani asks: "If ReWOO is that much cheaper, why keep ReAct at all?" Answer with one example from the
   lab where adapting to an observation matters.
2. Explain to Noor why Reflexion's evaluator is a security-relevant component, using the injected forum
   post as your example.

## 🔑 Key ideas

- ReWOO plans every tool call up front, so it needs two model calls regardless of the number of steps.
- Plans cannot adapt; plan-and-execute adds a re-planner that runs only when facts are missing.
- Reflexion turns detected failures into written lessons that the next attempt reads.
- An agent can only learn from failures its run-time evaluator can see; never evaluate with gold answers.
- Compare designs on the same tasks with the same metrics, on clean and dirty inputs, and include cost.

## 📚 Go deeper

- ⭐ [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) —
  the evaluator, self-reflection and memory loop, with experiments on coding and decision-making tasks.
- [ReWOO](https://arxiv.org/abs/2305.18323) — planner, worker and solver, and why decoupling them saves
  tokens.
- [Self-Refine](https://arxiv.org/abs/2303.17651) — a related idea: the model critiques and rewrites its
  own output within a single attempt.
- [ReAct](https://arxiv.org/abs/2210.03629) — the baseline every design in this lesson is compared with.

---

[← Previous: 8.1 Planning and the ReAct loop](01-planning-and-react.md) · [🏠 Module 08](README.md) · [Next: 8.3 Tree search for agents →](03-tree-search-for-agents.md)
