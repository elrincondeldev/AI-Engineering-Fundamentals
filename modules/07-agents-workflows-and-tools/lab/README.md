🌐 English · [Español](README.es.md)

# Lab 07 — Workflows, a guarded toolbox and an MCP server for Hearthwise Assist

> Build workflow patterns, a validated and permissioned toolbox over Hearthwise data, and a teaching
> MCP server and host that reuse the same tools. ⏱️ about 3 hours.

## 🎯 What you will build

| Piece | File | What it does |
|---|---|---|
| Agency ladder | `agency.py` | One request solved at four levels of autonomy, with cost and risk per level |
| Workflow patterns | `workflows.py`, `workflow_model.py` | Chaining with gates, routing, sectioning, voting, reflection, orchestrator-worker |
| Toolbox | `tools.py`, `kb.py`, `store.py` | `get_product`, `search_help_centre`, `lookup_order`, `create_return`, each with a JSON Schema |
| Validator | `schema.py` | A JSON Schema subset validator in plain Python |
| Executor | `executor.py` | Permissions, validation, human approval, timeouts, retries, idempotency keys, audit log |
| Tool loop | `loop.py`, `scripted.py` | The tool-calling loop and a deterministic scripted model that emits `ToolCall` objects |
| MCP server and host | `mcp_server.py`, `mcp_client.py` | A teaching JSON-RPC 2.0 server over stdio and a host with description linting and pinning |

Everything runs offline on a CPU with the `FakeProvider`. The order store is synthetic and generated
inside the lab with a fixed seed: five demo orders plus 55 random ones, all priced from the published
product pages. The lab's "today" is fixed at 2025-11-14 so that return windows never change.

> The MCP server and client here are a **teaching implementation** of a small subset of the Model
> Context Protocol (`initialize`, `notifications/initialized`, `tools/list`, `tools/call`). They are not
> the official SDK. Read the specification at https://modelcontextprotocol.io before building a real one.

## ⚙️ Setup

```bash
cd modules/07-agents-workflows-and-tools/lab
uv sync
uv run pytest
```

Expected test summary (the time is an example and varies by machine):

```text
93 passed in 3.84s
```

`uv run lab --help` lists the commands: `ladder`, `workflow`, `tools`, `mcp-demo` and `mcp-server`.

## 👣 Steps

### Step 1 — Install and run the tests

Run the setup commands above. All tests must pass before you change anything. The tests block network
access; the MCP tests start the server as a subprocess and talk to it over pipes.

```bash
uv run lab --help
```

```text
usage: lab [-h] {ladder,workflow,tools,mcp-demo,mcp-server} ...

Module 07 lab: workflows, tools and MCP.

positional arguments:
  {ladder,workflow,tools,mcp-demo,mcp-server}
    ladder              one request at four levels of agency
    workflow            run a workflow pattern
    tools               list the toolbox, or run a tool-calling demo
    mcp-demo            launch the teaching MCP server and talk to it
    mcp-server          run the teaching MCP server on stdin/stdout

options:
  -h, --help            show this help message and exit
```

### Step 2 — Climb the agency ladder

```bash
uv run lab ladder
```

```text
Request: I bought an Ember T3 (order HW-310042) and changed my mind. Please start a return.

Level                         Decides     LLM calls  Tools  Tokens  Data changed
1 single LLM call             nobody              1      0      63  no
2 workflow                    your code           1      1     154  no
3 tool-using (read-only)      the model           2      1    1416  no
4 agent (can act, approved)   the model           3      2    3314  yes
```

Read `agency.py`. Levels 3 and 4 call the same `_loop_rung`; the only difference is
`{"create_return": Permission.DENY}`. Lesson: [7.1](../01-agents-overview.md).

### Step 3 — Run the workflow patterns

```bash
uv run lab workflow chain
uv run lab workflow route
uv run lab workflow section
uv run lab workflow vote
uv run lab workflow reflect
uv run lab workflow orchestrate
```

`uv run lab workflow all` runs them in one go. The section pattern prints:

```text
== section ==
| Device | Price | Wi-Fi | Needs hub |
|---|---|---|---|
| Ember T3 | €149 | 2.4 GHz | no |
| Glow A19 | €12 | 2.4 GHz | no |
| Spark Mini | €19 | 2.4 GHz | no |
| Watcher S1 | €29 | none (HomeMesh radio) | yes |
| Lookout C2 | €89 | 2.4 GHz and 5 GHz | no |
| Hearth Hub | €69 | 2.4 GHz plus Ethernet port | no |
Sections: 6 | sequential rounds: 2 (would be 7 one after another)
LLM calls: 7 | sequential rounds: 2 | tokens: 1441
```

Lesson [7.2](../02-workflow-patterns.md) shows and explains the output of every pattern. Each step of
every workflow is a call to the same `FakeProvider`, whose responder (`workflow_model`) reads the task
name from the first line of the system prompt.

### Step 4 — Add a gate to the chain

The chain asks message 5 ("Return order 310155 please.") for its order number, although the customer
gave it, only without the `HW-` prefix. Fix it **in code**, before gate 1, in `chain_one` in
`workflows.py`: if the extracted id is exactly six digits, prefix it with `HW-`. Then run the chain:

```bash
uv run lab workflow chain
```

Expected last lines after your change:

```text
Outcomes: {'replied': 4, 'asked_for_order_id': 1, 'not_found': 1}
LLM calls: 10 | sequential rounds: 2 | tokens: 921
```

Message 5 now gets "Your order HW-310155 can be returned: 19 days left. ...". Add a test for it in a
new file `tests/test_my_gate.py`. `test_chain_gate_stops_early_without_a_valid_order_id` in
`tests/test_workflows.py` and the documented chain numbers in `tests/test_documented_numbers.py` will
now fail: update the first, and keep the second as a reminder that documented numbers are tied to code
(or put your change on a branch).

<details><summary>Solution</summary>

```python
fields = json.loads(meter.ask("extract", message))
order_id = fields.get("order_id")
if order_id and re.fullmatch(r"\d{6}", order_id):
    order_id = f"HW-{order_id}"  # normalise before the gate: cheap, deterministic, testable
if not order_id or not re.match(ORDER_PATTERN, order_id):  # gate 1: stop early, ask the customer
    ...
```

```python
from m07_agents_tools import workflows as wf


def test_prefixless_order_id_is_normalised() -> None:
    result = wf.chain_one(wf.Meter(), wf.OrderStore(), "Return order 310155 please.")
    assert result["outcome"] == "replied" and "HW-310155" in result["reply"]
```

</details>

### Step 5 — Inspect the toolbox and the tool loop

```bash
uv run lab tools
uv run lab tools --demo fix-args
```

```text
Tool                Changes data  Required arguments                  Spec tokens
get_product         no            product                                     192
search_help_centre  no            query                                       194
lookup_order        no            order_id                                    150
create_return       yes           order_id, sku, quantity, reason             331
All four specs together: 872 tokens, sent with every LLM call.
== fix-args ==
Request: Where is my order 310108?
  LLM call 1 -> lookup_order({"order_id": "310108"})
      invalid_arguments $.order_id: '310108' does not match pattern ^HW-\d{6}$. Fix the arguments and call the tool a...
  LLM call 2 -> lookup_order({"order_id": "HW-310108"})
      ok (1 attempt)
  LLM call 3 -> answer: Order HW-310108 is shipped.
  Totals: 3 LLM calls, 2 tool calls, 3073 prompt tokens, 29 completion tokens
  Returns in the store: 0
```

Open `scripted.py` and find the line that makes the model copy what the customer typed, and the line
that corrects it after reading the error. Then read `run_tool_loop` in `loop.py`, the whole loop in
one short function.
Lesson: [7.3](../03-tool-calling.md).

### Step 6 — Make tool execution safe

Run the demos that exercise approval, ownership, retries, timeouts and idempotency:

```bash
uv run lab tools --demo return
uv run lab tools --demo return-denied
uv run lab tools --demo not-yours
uv run lab tools --demo retries
uv run lab tools --demo timeout
uv run lab tools --demo idempotency
```

The `retries` demo injects two transient failures; the `timeout` demo makes a tool sleep 0.5 s against a
0.2 s timeout. Expected key lines:

```text
== retries ==
Request: Where is order HW-310108?
  LLM call 1 -> lookup_order({"order_id": "HW-310108"})
      ok (3 attempts)
```

```text
== not-yours ==
Request: Where is order HW-418230?
  LLM call 1 -> lookup_order({"order_id": "HW-418230"})
      error No order HW-418230 found for this customer.
  LLM call 2 -> answer: I could not find order HW-418230 in your account. Could you check the number?
  Totals: 2 LLM calls, 1 tool call, 1914 prompt tokens, 30 completion tokens
  Returns in the store: 0
```

In the `idempotency` demo, compare `Returns in the store: 1` (keys on) with `Returns in the store: 2`
(keys off). Then read `ToolExecutor._execute` and `_run_with_retries` in `executor.py` and answer: in
which order are the checks made, and which errors are never retried?

### Step 7 — Add a fifth tool

Add a read-only `check_warranty` tool that returns, for one delivered order of the signed-in customer,
the end of the standard 2-year warranty and of the 3-year warranty with Hearthwise Plus (see
`kb/policies/warranty.md`). You need three things: a `warranty()` method on `OrderStore`, a `Tool` entry
in `build_toolbox` with a description and a JSON Schema, and nothing else. Then check it three ways:

```bash
uv run lab tools          # a fifth row appears
uv run lab mcp-demo       # the server now lists five tools, with no protocol code changed
uv run python -c "
from m07_agents_tools.executor import ToolExecutor
from m07_agents_tools.providers import ToolCall
from m07_agents_tools.tools import Session, build_toolbox
ex = ToolExecutor(build_toolbox(), Session('C-0042'))
print(ex.execute(ToolCall('c1', 'check_warranty', {'order_id': 'HW-310042'})).data)
"
```

Expected output of the last command:

```text
{'order_id': 'HW-310042', 'delivered_on': '2025-10-23', 'standard_until': '2027-10-23', 'with_plus_until': '2028-10-23'}
```

Tests that pin the toolbox to four tools (in `test_mcp.py`, `test_cli.py` and
`test_documented_numbers.py`) will fail. That is the documented-numbers contract working: the lessons
quote four tools and 872 tokens. Keep your tool on a branch, or update the tests and the numbers
together. In this lab the host computes its approved pins from the same code, so it trusts your new tool
at once; in production the pins come from the last review, and `check_warranty` would be quarantined
until Noor approves it.

<details><summary>Solution</summary>

```python
# store.py, inside OrderStore
def warranty(self, order_id: str, customer_id: str) -> dict[str, object]:
    order = self._owned(order_id, customer_id)
    if order.delivered_on is None:
        raise ToolError(f"Order {order_id} has not been delivered yet.")
    d = order.delivered_on
    return {"order_id": order_id, "delivered_on": d.isoformat(),
            "standard_until": d.replace(year=d.year + 2).isoformat(),
            "with_plus_until": d.replace(year=d.year + 3).isoformat()}
```

```python
# tools.py, add to the list in build_toolbox
Tool(
    name="check_warranty",
    description=(
        "Get the warranty end dates of one delivered order of the signed-in customer: the standard "
        "2-year warranty and the 3-year warranty with Hearthwise Plus. Use it when a customer reports "
        "a faulty device. Read-only."
    ),
    parameters=_obj({"order_id": ORDER_ID}, ["order_id"]),
    handler=lambda a, s: store.warranty(a["order_id"], s.customer_id),
    timeout_s=timeout_s,
),
```

</details>

### Step 8 — Speak MCP by hand

Send JSON-RPC messages to the teaching server yourself, one per line:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"teaching-subset","capabilities":{},"clientInfo":{"name":"terminal","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  'this is not json' | uv run lab mcp-server
```

```text
{"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "Server not initialized: send initialize first"}}
{"jsonrpc": "2.0", "id": 2, "result": {"protocolVersion": "teaching-subset", "capabilities": {"tools": {}}, "serverInfo": {"name": "hearthwise-tools", "version": "0.1.0"}}}
{"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "Parse error"}}
hearthwise-tools: stdin closed, exiting
```

Four messages in, three responses out: the notification gets none. Now read `TeachingMcpServer.handle`
in `mcp_server.py` and find where each response came from. Lesson: [7.4](../04-model-context-protocol.md).

### Step 9 — Run the host and catch a poisoned server

```bash
uv run lab mcp-demo
uv run lab mcp-demo --poisoned
```

The first run shows the handshake, the host's review, two direct calls (one refused because the order
belongs to another customer) and the lesson 7.3 tool loop running over MCP, ending with
`JSON-RPC messages exchanged: 13`. The second run connects to a server with a poisoned `check_stock`
tool:

```text
  check_stock          findings: ['hidden-instruction markup', 'tells the model to hide something', 'mentions other tools: lookup_order']
  pin diff: {'added': ['check_stock'], 'removed': [], 'changed': []}
  exposed to the model: ['get_product', 'search_help_centre', 'lookup_order', 'create_return'] | quarantined: ['check_stock']
```

Read `remote_tools`, `lint_description`, `pin_tools` and `diff_pins` in `mcp_client.py`. Which of them
would still protect you if the poisoned tool had been there from the first review?

## 💥 Break it

**1. Remove the order id pattern.** Delete `"pattern"` from `ORDER_ID` in `tools.py` and run
`uv run lab tools --demo fix-args`. What does the customer hear about an order that exists?

<details><summary>What happens</summary>

The model's `"310108"` now passes validation, the store finds no such order, and the tool returns a
business error instead of `invalid_arguments`. The model cannot know the argument was malformed, so it
tells the customer "I could not find order HW-310108 in your account" about an order that is on its way.
A precise schema turns a misleading answer into a self-correction.

</details>

**2. Switch off idempotency keys.** In `run_demo` in `cli.py`, change the default `keys: bool = True`
to `False` and run `uv run lab tools --demo idempotency`.

<details><summary>What happens</summary>

Both halves now print `Returns in the store: 2`. The customer was told "Done" once and has two open
returns, which means two refunds. The timeout hid the fact that the first call had already succeeded.

</details>

**3. Starve the loop.** Call `run_tool_loop(..., max_steps=1)` in `run_demo` and run the `return` demo.

<details><summary>What happens</summary>

The model's first call is `lookup_order`; the loop then stops with `stop_reason == "max_steps"` and an
empty answer. Budgets protect you from runaway loops, but a budget that is too small fails every
legitimate multi-step request. Log the stop reason and alert on `max_steps`.

</details>

**4. Let hidden HTML through.** In `_TextExtractor.handle_starttag` in `kb.py`, make `hidden` always
`False`, then run
`uv run python -c "from m07_agents_tools import kb; print(kb.search('refund verify')[0]['id'])"`.

<details><summary>What happens</summary>

It prints `kb-025`: the community page now ranks first, and its snippet is the hidden paragraph aimed at
AI assistants. A search tool's output goes straight into the model's context, so whatever a page hides
becomes part of the prompt. Dropping hidden elements is one layer; treating results as data is the other.

</details>

**5. Trust the server's word.** In `remote_tools` in `mcp_client.py`, set `side_effect=False` for every
tool and run `uv run lab mcp-demo`.

<details><summary>What happens</summary>

The `[approval]` line disappears: `create_return` runs over MCP without a person approving it. The host
just let a remote server decide what counts as harmless. Keep the write-tool list on the host.

</details>

## 🚀 Extend it

**1. Cap the orchestrator.** Add `max_rounds: int = 3` to `orchestrate` and stop adding workers when it
is reached; record `"capped": True` in the report.

<details><summary>Hint</summary>

Count iterations of the `while todo:` loop. When the cap is reached, break before calling `replan` again
and still run `synthesise` on what you have. Test it with a fake replan that always asks for one more
product.

</details>

**2. Do not retry ambiguous side effects.** When a side-effecting tool times out and no idempotency key
is attached, return a `timeout` result with the message "Outcome unknown: check the order before trying
again" instead of retrying.

<details><summary>Hint</summary>

In `_run_with_retries`, pass in whether a key is attached. On `FutureTimeout`, if
`tool.side_effect and not has_key`, return immediately. With Break it 2 applied, the idempotency demo
should now leave one return in the store and tell the customer the outcome is unknown.

</details>

**3. Add resources to the teaching server.** Offer the current returns policy (kb-011) as an MCP
resource with `resources/list` and `resources/read`, and advertise the capability in `initialize`.

<details><summary>Hint</summary>

Check the exact request and result shapes in the resources section of the specification at
https://modelcontextprotocol.io before you write code. Return the document's text, not the HTML. Keep
`resources/read` limited to public, current documents; the internal playbook must never be listed.

</details>

**4. Give the loop a token budget.** Add `max_prompt_tokens` to `run_tool_loop`. When a model call asks
for tools and the running prompt-token total is already above the budget, stop with
`stop_reason = "budget"` *before* running those tools.

<details><summary>Hint</summary>

The completion carries `prompt_tokens`; add it up after each call. In the `compare` demo the two calls
cost 932 and 1,343 prompt tokens; in the `return` demo the three calls cost 937, 1,079 and 1,198. With a
budget of 2,000, `compare` still answers, while `return` stops at a running total of 2,016, before
`create_return` runs. Checking before tools run means a budget can never interrupt a side effect halfway.

</details>

**5. Persist the pins.** Save the approved pins to `lab/outputs/approved_pins.json` the first time
`mcp-demo` runs, and load them on later runs, so that a tool you add in step 7 is quarantined until you
delete the file.

<details><summary>Hint</summary>

Use `pathlib.Path("outputs")`, `mkdir(exist_ok=True)` and `json.dumps(pins, indent=2)`. Load the file if
it exists; otherwise compute the pins from the live `tools/list` and write them. `outputs/` is ignored
by Git.

</details>

## ✅ Definition of Done

- [ ] `uv run pytest` passes on a clean checkout.
- [ ] You ran `ladder`, all six `workflow` patterns, every `tools --demo` and both `mcp-demo` variants.
- [ ] Step 4: prefixless order ids are normalised before gate 1, with a test.
- [ ] Step 7: `check_warranty` exists, validates its input, uses the session's customer id and appears
      over MCP without protocol changes.
- [ ] You can explain each executor status (`invalid_arguments`, `denied`, `rejected`, `timeout`, `error`)
      and whether the model should retry it.
- [ ] You did at least three Break it experiments and wrote one sentence about each.
- [ ] You completed at least one Extend it item, with a test.

## 🧯 Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` | Install uv from https://docs.astral.sh/uv/ and open a new terminal |
| `No interpreter found for Python 3.12` | `uv python install 3.12`, then `uv sync` again |
| `Could not find the repo's datasets/ folder` | Run from inside the repository, or set `COURSE_DATASETS_DIR` to the `datasets` path |
| `McpError: initialize: no response within 10.0 s` | The server crashed at start-up. Run `uv run lab mcp-server < /dev/null` to see its error on standard error |
| Tests fail after step 4 or step 7 | Expected: some tests pin the documented numbers. See the notes in those steps |
| `error: Cannot reach http://localhost:11434/api/chat` | You used `--live` with `LLM_PROVIDER=ollama` and Ollama is not running |

## 🔌 Using a real model

Every command above uses the scripted model so that results are reproducible. The tool loop itself is
provider-agnostic: `run_tool_loop` sends `executor.specs()` (OpenAI-style function definitions) through
the shared provider layer in `providers.py`, which serialises tool calls and tool results for Ollama and
for OpenAI-compatible endpoints.

```bash
# Ollama: start the server, pull a model that supports tool calling, then:
LLM_PROVIDER=ollama OLLAMA_MODEL=<a tool-capable model> uv run lab tools --demo compare --live

# Any OpenAI-compatible endpoint:
LLM_PROVIDER=openai OPENAI_BASE_URL=<url> OPENAI_API_KEY=<key> OPENAI_MODEL=<model> \
  uv run lab tools --demo compare --live
```

With `--live`, the demos use `get_provider()` instead of the scripted model. Expect different traces: a
real model may choose other tools, phrase arguments differently or answer without tools. The executor's
guarantees (validation, permissions, approval, idempotency) hold whatever the model does, which is the
point of putting them in code.

**What was not tested:** the `--live` path was not run against a real model while writing this lab.
Ollama was installed but not running in the authoring environment, and no API key was used. The tests
only check, offline, that the provider layer builds correct request payloads and that `--live` reports
a clear error when no server is reachable. Live tests would be marked `@pytest.mark.ollama` or
`@pytest.mark.openai` and run with `RUN_LIVE_TESTS=1`.

---

[← Previous: 7.4 The Model Context Protocol](../04-model-context-protocol.md) · [🏠 Module 07](../README.md) · [Next: Quiz →](../quiz.md)
