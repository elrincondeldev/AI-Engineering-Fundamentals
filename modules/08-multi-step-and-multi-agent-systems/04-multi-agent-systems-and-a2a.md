🌐 English · [Español](04-multi-agent-systems-and-a2a.es.md)

# Lesson 8.4 — Multi-agent systems and A2A

> **Module 08 · Lesson 4 of 5** · ⏱️ about 60 min · 🧪 Lab: [step 7](lab/README.md#step-7--run-the-multi-agent-system)

## 📨 Situation

> **From:** Noor Haddad · Security and Privacy Lead
> **Channel:** #ask-the-web
>
> Jordan forwarded me a slide that splits the research assistant into "a team of agents": a supervisor,
> a web researcher and a fact-checker, talking over something called Agent2Agent (A2A). Before anyone
> builds it, I have two questions. First, what is the blast radius when one of those agents reads a
> poisoned page: does the injected text stay inside the researcher, or does it travel to every agent
> downstream? Second, how is A2A different from the Model Context Protocol (MCP) server you built in
> module 07? I would like numbers and a diagram, not a slide.

## 🎯 Objectives

By the end of this lesson you can:

- Decide when several agents beat one (specialisation, parallelism, isolation) and when they do not.
- Compare supervisor, hierarchical and debate or critic topologies.
- Measure coordination overhead (model calls, tokens, duplicated tool calls) with the lab.
- Explain how errors and injected text propagate between agents, and how structured handoffs contain them.
- Contrast the Agent2Agent (A2A) protocol with the Model Context Protocol (MCP).

## 🤔 Predict first

Write your answers down before reading on. You will check them in the self-check.

1. The single ReAct agent averages 4.25 model calls per task. Guess the average for a supervisor that
   delegates to a researcher and sends every claim to a fact-checker.
2. With every guard off, will adding the fact-checker fix more tasks than it breaks?

## 1. Why use more than one agent?

A multi-agent system splits one job across several model-driven components, each with its own prompt,
tools and context. Three reasons justify the split:

| Reason | What it buys | Hearthwise example |
|---|---|---|
| Specialisation | Shorter, focused prompts and tool sets per role | A researcher that only searches; a checker that only verifies |
| Parallelism | Independent sub-questions run at the same time | The two prices in task06 can be fetched concurrently |
| Isolation | A component that reads untrusted text holds no dangerous tools | The researcher reads the web but cannot send e-mail |

And three costs come with it, every time:

| Cost | Why it happens |
|---|---|
| Coordination | Every handoff is extra model calls and extra tokens to describe the work |
| Error compounding | Agent B trusts agent A's output; a mistake upstream is a premise downstream |
| Duplicated work | Agents that do not share memory repeat searches and page reads |

Frameworks such as [AutoGen](https://arxiv.org/abs/2308.08155) make multi-agent conversations easy to
build. Easy to build is not the same as worth building: the question is always whether the benefits
above show up in *your* evaluation.

**Analogy:** a newsroom. A reporter gathers facts, a fact-checker verifies them, an editor decides what
runs. It works because each role is clear and handoffs are written down. It also means a story takes
longer and costs more than one person writing a blog post.

> ⚠️ **Common mistake:** splitting by job titles instead of by context. If two agents need the same
> pages and the same history to do their work, they are one agent with extra overhead.

## 2. Topologies: who talks to whom

The shape of the conversation matters as much as the agents in it.

| Topology | How it works | Good for | Watch out for |
|---|---|---|---|
| Supervisor | One agent plans, delegates sub-tasks and writes the answer | Decomposable questions | The supervisor is a bottleneck and a single point of failure |
| Hierarchical | Supervisors of supervisors | Large, multi-domain jobs | Each layer adds latency and lossy summaries |
| Debate or critic | Agents propose and challenge each other's answers | Catching unsupported claims | Agreement is not truth; critics can be wrong too |
| Network | Any agent can call any other | Open ecosystems | Hard to bound cost and to trace |

A note on debate and critics: two models that agree are not two witnesses if they read the same pages.
A critic adds value only when it has a different view of the evidence (another source, a stricter rule,
a different model), and it should be able to say "I cannot verify this" rather than being forced to
pick a side.

The lab builds a **supervisor with a critic**:

```mermaid
flowchart TD
    U["Question"] --> S["Supervisor<br/>plans fact groups, writes answer"]
    S -->|"task: find facts (structured slots)"| R["Researcher<br/>ReAct over search_web and open_page"]
    R -->|"artifact: values, URLs, dates"| S
    S -->|"task: check these claims"| C["Fact-checker<br/>reads up to 2 other results"]
    C -->|"artifact: confirmed, unverified or overruled"| S
    S --> A["Final answer with sources"]
```

The supervisor groups facts that share a search query (the five countries of a market table become one
task), sends each group to the researcher, sends the researcher's claims to the fact-checker, and
composes the answer from the facts that survive. Here is the traffic for the adoption comparison
(task05):

```text
card: researcher · skills ['web_research'] · Finds one fact on the offline web and cites it.
card: fact_checker · skills ['fact_check'] · Checks a claimed fact against a second source.
-> #1 send_message to researcher: [slot: adoption | Mexico | smart home adoption households Mexico Chile 2025] [slot: adoption | Chile | smart home adoption households Mexico Chile 2025]
<- #1 task-1 state=completed
-> #2 send_message to fact_checker: 2 claim(s)
<- #2 task-2 state=completed
```

> 💡 **Production tip:** start with a supervisor and one specialist. Add a role only when the evaluation
> shows a failure the new role fixes, and measure what it costs in the same run.

## 3. What coordination costs

Run the same 24 tasks through the single agent and the multi-agent system, with every guard on:

| Design | Success | Avg model calls | Avg tokens | Avg tool calls | Redundant tool calls (total) |
|---|---|---|---|---|---|
| ReAct (one agent) | 24/24 | 4.25 | 3,244 | 3.25 | 0 |
| Supervisor + researcher | 24/24 | 6.79 | 3,498 | 3.50 | 3 |
| Supervisor + researcher + fact-checker | 24/24 | 8.00 | 4,211 | 7.12 | 54 |

Same accuracy, roughly double the model calls once the fact-checker joins. The supervisor's own calls
(planning and writing) add overhead even without the critic. The fact-checker repeats the researcher's
search and often re-opens pages the researcher already read, which is where the 54 redundant calls come
from: two agents that do not share a cache do the same work twice. On task06, the system spent 12 model
calls, 12 tool calls and 6,954 tokens to answer what one agent answered with 7 model calls and 6 tool
calls.

**Analogy:** a meeting. Every extra attendee costs everyone's time, so you invite someone only if the
meeting would decide worse without them.

```mermaid
sequenceDiagram
    participant S as Supervisor
    participant R as Researcher
    participant C as Fact-checker
    S->>R: find price of Corvid Thermo X
    R-->>S: €179, corvid review (updated)
    S->>C: check €179
    C-->>S: confirmed
    S->>R: find price of Ember T3
    R-->>S: €149, Ember T3 review
    S->>C: check €149
    C-->>S: confirmed
```

> 💡 **Production tip:** give agents a shared, read-only cache of tool results. Redundant page reads are
> pure waste, and in a live system they also hit rate limits on the sites you read.

## 4. Errors compound, and so does injected text

Now switch every guard off and watch what the extra agents do with dirty evidence:

| Variant (all guards off) | Success | Failed | What happened |
|---|---|---|---|
| Single ReAct agent | 22/24 | task04, task19 | Believed the spam page; repeated the injected advert |
| Multi-agent, no fact-checker | 23/24 | task04 | Same spam error; the advert stayed inside the researcher |
| Multi-agent with fact-checker | 23/24 | task02 | Checker rejected the spam page, but overruled a correct answer |
| Multi-agent, free-text handoff | 22/24 | task02, task19 | The advert travelled to the final answer |

Two effects deserve a close look.

**The critic is a new source of errors.** On task02 the researcher correctly found 250 devices in the
HomeMesh specification. The fact-checker searched again, read the forum post and its tracking-parameter
copy, saw two "sources" agreeing on 200, and overruled the researcher. A downstream agent that can
overwrite upstream results can also overwrite them wrongly; its errors are as real as the ones it fixes.

**Handoffs decide the blast radius.** The researcher that reads the injected forum post is the same
gullible ReAct model as before, and its free-text answer ends with the competitor advert. What happens
next depends on what the supervisor accepts:

```python
found = task.artifacts[0].parts[0].content      # data part: {slot: {value, url, published, ...}}
if not self.structured_handoff:                 # free-text handoff: keep what the researcher wrote
    notes.append(str(task.artifacts[0].parts[1].content))
for slot in slots:
    if slot.key in found:                       # structured handoff: typed fields only
        chosen[slot.key] = _fact(slot, found[slot.key])
```

With the structured handoff, the supervisor reads only typed fields (a value, a web address or URL, a date), so the
advert never reaches the final answer, even with the sanitiser off. With the free-text handoff, the
writer pastes the researcher's prose and the advert ships to the user. Answering Noor: the blast radius
of a poisoned page is every agent that consumes free text derived from it.

**Analogy:** a game of telephone versus a form. A whispered message gets distorted and can be hijacked
at every hop; a form with fixed fields can only carry what the fields allow.

> ⚠️ **Common mistake:** passing one agent's raw output into another agent's prompt "for context". That
> turns every downstream agent into a reader of untrusted text. Hand off typed data, validate it, and
> give the agents that read the web the fewest privileges.

## 5. A2A and MCP: two different connections

In module 07 you exposed tools through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io):
a standard way for an agent to reach **tools and data**. The [Agent2Agent (A2A)
protocol](https://github.com/a2aproject/A2A) addresses a different connection: an agent asking **another
agent**, possibly built by another team or vendor, to do a piece of work. Its core concepts are:

- **Agent card:** a description an agent publishes (name, what it does, its skills, where to reach it)
  so that clients can discover it.
- **Task:** a unit of work with a lifecycle; states include submitted, working, input-required,
  completed and failed.
- **Messages and artifacts:** messages carry the conversation as parts (text or structured data);
  artifacts carry the results.
- **Transport:** the web's Hypertext Transfer Protocol (HTTP), carrying JSON Remote Procedure Call
  (JSON-RPC) 2.0 messages.

| | MCP | A2A |
|---|---|---|
| Connects | an agent to tools, resources and prompts | an agent to another agent |
| The other side is | a capability you call directly | a peer that plans and acts on its own |
| Typical unit | a tool call with arguments and a result | a task that moves through states and returns artifacts |
| Hearthwise example | `search_web` exposed as an MCP tool | the fact-checker offered to other teams' agents |

Why a *task* with states rather than a simple function call? Because the other side is an agent. Its
work can take minutes, it may need to ask a question back ("which market do you mean?", the
input-required state), it may fail halfway, and the caller may want to cancel it. A function call has
no vocabulary for any of that. The other important property is **opacity**: an A2A client does not see
the remote agent's prompts, tools or memory, only its card, its messages and its artifacts. That is what
lets two teams, or two companies, connect agents without sharing their internals, and it is also why you
must validate everything that comes back.

The lab simulates these ideas in-process; nothing opens a socket. The researcher and fact-checker
publish `AgentCard`s, the supervisor discovers them by skill, every request creates a `Task` whose state
moves from submitted to working to completed, and each exchange is logged as a JSON-RPC 2.0 envelope.
The method name in the log (`send_message`) is the lab's own label; real method names and fields are in
the specification.

```python
def send(self, card: AgentCard, message: A2AMessage) -> Task:
    n = next(self._ids)
    task = Task(id=f"task-{n}", agent=card.name, messages=[message])
    self.wire.append(
        jsonrpc_envelope("send_message", {"agent": card.name, "message": asdict(message)}, n)
    )
    task.move("working")
    self.handlers[card.name](task)              # the remote agent does its work
    self.wire.append({"jsonrpc": "2.0", "id": n, "result": {"task": task.id, "state": task.state}})
    return task
```

```mermaid
sequenceDiagram
    participant S as Supervisor (client)
    participant D as Directory
    participant C as Fact-checker (remote agent)
    S->>D: discover skill "fact_check"
    D-->>S: agent card
    S->>C: JSON-RPC request with a message (data part: claims)
    Note over C: task: submitted, then working
    C-->>S: task completed, artifact: verdicts
```

**Analogy:** MCP is the socket on the wall that any appliance can plug into. A2A is a phone call to a
contractor: you describe the job, they work on it in their own way, and they report back when it is
done or when they need something from you.

> 💡 **Production tip:** treat a remote agent like any third-party application programming interface (API) that returns untrusted content:
> authenticate it, set timeouts and budgets per task, validate its artifacts against a schema, and log
> every task with its states.

## 🧪 Try it

Run from the lab folder.

1. Watch the agents talk:

   ```bash
   uv run lab task task05 --agent multi --trace
   ```

   ```text
   Final Answer: Chile
   Explanation: Smart home adoption in Mexico: 14% [1]; smart home adoption in Chile: 16% [1].
   Sources: [1] https://marketlens.example/reports/smart-home-adoption-2025
   Cost: 6 model calls · 5 tool calls · 2,755 tokens (prompt 2,485 + completion 270)
   ```

2. Measure coordination overhead with and without the critic:

   ```bash
   uv run lab eval --agent multi
   uv run lab eval --agent multi --no-critic
   ```

   ```text
   Avg tool calls: 7.12 · avg model calls: 8.00 · avg tokens: 4,211
   Tool errors: 0 · redundant calls: 54 · must_not_contain violations: 0 · uncited domains: 0
   ```

   (first command; the second prints 3.50 tool calls, 6.79 model calls and 3 redundant calls)

3. See the critic overrule a correct answer:

   ```bash
   uv run lab task task02 --agent multi --naive --trace
   ```

   The notes include `devices:HomeMesh 2.0: overruled` and the answer is 200.

4. Let injected text travel between agents:

   ```bash
   uv run lab task task19 --agent multi --naive --free-text-handoff
   ```

   ```text
   Expected: Use a dedicated 2.4 GHz network and fix the channel (1, 6 or 11). · correct: True · exact: False · violations: ['Corvid', 'gadgetdeals'] · citations ok: 1/1
   ```

## 🏭 In the real world

Multi-agent designs pay off when the parts are genuinely separate: different data access (an agent
inside the finance system, another on the public web), different owners (a partner's agent you cannot
modify), or work that truly runs in parallel. They disappoint when used to make a single task feel
organised: the bill and the latency grow, and debugging means reading several transcripts that refer
to each other.

In production, trace every task end to end with one correlation identifier (ID), cap budgets per agent and per task,
cache shared tool results, and define the schema of every handoff. Security reviews should treat each
agent as a separate principal with its own permissions; the agent that reads untrusted pages should be
the one with the fewest privileges, and nothing it writes should be executed as an instruction.

## ✅ Self-check

**1. How many model calls per task does the supervisor-plus-two-specialists system make, compared with
4.25 for ReAct?** (Predict-first question 1.)

<details><summary>Answer</summary>

8.00 with every guard on, for the same 24/24 success. The supervisor's planning and writing calls, the
researcher's ReAct calls and the fact-checker's verdicts all add up. Without the fact-checker it is 6.79.

</details>

**2. With every guard off, does the fact-checker fix more than it breaks?** (Predict-first question 2.)

<details><summary>Answer</summary>

It fixes one and breaks one: it rejects the spam page on task04, but overrules the correct 250 on task02
because two copies of a forum post agree on 200. Success stays at 23/24 either way, so on this suite the
critic costs extra calls without a net gain.

</details>

**3. Why did the injected advert reach the final answer only with the free-text handoff?**

<details><summary>Answer</summary>

Because the supervisor's writer then read the researcher's prose, which repeated the advert. With the
structured handoff it reads only values, URLs and dates, so text from the page cannot travel downstream.

</details>

**4. Is A2A a replacement for MCP?**

<details><summary>Answer</summary>

No. MCP connects an agent to tools and data; A2A connects an agent to another agent that works on a task
on its own and reports back through task states and artifacts. A system can use both.

</details>

**5. Where do the 54 redundant tool calls come from, and how would you remove most of them?**

<details><summary>Answer</summary>

The fact-checker repeats the researcher's search and re-opens pages the researcher already read. A shared
read-only cache of tool results, keyed by canonical URL and normalised query, removes most of them.

</details>

## 💬 Explain it in your own words

1. Answer Noor's first question in five sentences: what is the blast radius of a poisoned page in the
   lab's multi-agent system, and which design choice controls it?
2. Explain to Jordan, without jargon, why "a team of agents" made the assistant slower and more
   expensive without making it more accurate on the 24 tasks.

## 🔑 Key ideas

- Split into several agents for specialisation, parallelism or isolation, and only when the evaluation
  shows a gain.
- Coordination costs model calls, tokens and duplicated tool calls; measure them on the same tasks.
- Downstream agents inherit upstream errors, and a critic can introduce errors of its own.
- Structured handoffs (typed values, URLs, dates) contain injected text; free-text handoffs spread it.
- MCP connects agents to tools and data; A2A connects agents to agents through cards, tasks, messages
  and artifacts over HTTP and JSON-RPC.

## 📚 Go deeper

- ⭐ [Agent2Agent (A2A) protocol](https://github.com/a2aproject/A2A) — the specification, concepts and
  samples; read the agent card and task lifecycle sections first.
- [Model Context Protocol](https://modelcontextprotocol.io) — the tool side of the picture, revisited
  from module 07.
- [AutoGen](https://arxiv.org/abs/2308.08155) — a framework and paper on multi-agent conversations, with
  examples of different topologies.
- [Indirect prompt injection](https://arxiv.org/abs/2302.12173) — why every agent that reads untrusted
  content needs containment.

---

[← Previous: 8.3 Tree search for agents](03-tree-search-for-agents.md) · [🏠 Module 08](README.md) · [Next: 8.5 Evaluating agents →](05-evaluating-agents.md)
