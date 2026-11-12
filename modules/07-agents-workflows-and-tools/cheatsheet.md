🌐 English · [Español](cheatsheet.es.md)

# Module 07 cheatsheet — Workflows, tools and MCP

## Key definitions

| Term | One line |
|---|---|
| LLM call | One prompt in, one completion out; nobody decides a next step |
| Workflow | Your code defines the steps; LLM calls fill them in |
| Agent | The model decides the next step (which tool, when to stop) inside limits your code enforces |
| Agentic system | Umbrella term for workflows and agents |
| Tool call | A structured request from the model (`id`, `name`, JSON `arguments`); your code runs it |
| Idempotency key | A key that makes repeating an action harmless: same key, same result, no second action |
| MCP | Model Context Protocol: a standard for AI applications to discover and call tools on servers |
| Host / client / server | App the person uses / connector inside it, one per server / program offering tools |

## The agency ladder (lab: `uv run lab ladder`)

| Level | Who decides | LLM calls | Tokens | Data changed |
|---|---|---|---|---|
| 1 Single LLM call | nobody | 1 | 63 | no |
| 2 Workflow | your code | 1 | 154 | no |
| 3 Tool-using (read-only) | the model | 2 | 1,416 | no |
| 4 Agent (can act, approved) | the model | 3 | 3,314 | yes |
| 5 Multi-agent | several models | module 08 | | |

Do **not** build an agent when the steps are known, a button already exists, a wrong action is costly
to undo, you cannot evaluate it, or the latency budget is tight.

## Workflow patterns (lab: `uv run lab workflow <pattern>`)

| Pattern | Use when | Lab result |
|---|---|---|
| Chaining + gates | Known, checkable steps | 9 LLM calls for 6 messages instead of 12 |
| Routing | Distinct input types | 0.848 accuracy, 0.954 when routed, 136 of 1,224 to people |
| Sectioning | Independent parts | 7 calls, 2 sequential rounds instead of 7 |
| Voting | Random slips, high stakes | k=1/3/5: 0.613 / 0.697 / 0.770; ceiling 0.848 |
| Reflection | Evidence or rubric exists | 3 issues in round 1, passed in round 2 |
| Orchestrator-worker | Subtasks depend on findings | Hub added after a worker reported `needs_hub` |

Latency follows **sequential rounds**; cost follows **LLM calls and tokens**.

## A good tool definition

- Name: verb + noun (`lookup_order`).
- Description: what it does, when to use it, when not to, what it returns, units, "changes data".
- Schema: `type`, `enum` for closed lists, `pattern` for formats, `minimum`/`maximum`, `required`,
  `"additionalProperties": false`.
- Never a parameter: the user's identity. Take it from the session.
- Every definition is sent with every call: the lab's four cost 872 tokens.

## The executor's order of checks

```text
exists? -> permission (allow / approve / deny) -> validate schema -> human approval (side effects)
       -> run with timeout -> retry transient errors with backoff (same idempotency key) -> audit log
```

| Status | Model should... |
|---|---|
| `invalid_arguments` | fix the arguments and call again |
| `unknown_tool` | pick a listed tool |
| `denied` / `rejected` | not retry; explain or hand over |
| `timeout` / `error` | usually not retry; explain or escalate |

## MCP in one screen

```text
-> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{protocolVersion, capabilities, clientInfo}}
<- {"jsonrpc":"2.0","id":1,"result":{protocolVersion, capabilities, serverInfo}}
-> {"jsonrpc":"2.0","method":"notifications/initialized"}            (notification: no id, no reply)
-> {"jsonrpc":"2.0","id":2,"method":"tools/list"}
<- {"jsonrpc":"2.0","id":2,"result":{"tools":[{name, description, inputSchema}]}}
-> {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":..., "arguments":{...}}}
<- {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":...}],"isError":false}}
```

- Primitives: **tools** (model decides), **resources** (application decides), **prompts** (person decides).
- Transports: **stdio** (subprocess, one JSON message per line, logs on stderr) and HTTP-based (remote).
- Protocol problems: JSON-RPC errors (-32700 parse, -32600 invalid request, -32601 method not found,
  -32602 invalid params). Tool failures: `result` with `isError: true`.

## MCP security

| Risk | Mitigation |
|---|---|
| Tool poisoning (instructions in descriptions) | Lint descriptions, review, pin definitions, quarantine new or changed tools |
| Over-broad permissions | Narrow credentials per server; deny write tools for read-only hosts |
| Confused deputy | Identity from the session or launch environment, enforced by the server |
| Untrusted servers | Review source, pin versions, minimal environment, treat results as data |

## Lab commands

```bash
uv run lab ladder                      # one request, four levels of agency
uv run lab workflow all                # chain, route, section, vote, reflect, orchestrate
uv run lab tools                       # tool table and spec token costs
uv run lab tools --demo all            # fix-args, compare, approval, retries, timeout, idempotency
uv run lab mcp-demo [--poisoned]       # host + teaching MCP server over stdio
uv run lab mcp-server                  # the server alone, on stdin/stdout
```

## Three most common mistakes

1. **Safety rules only in the prompt.** Enforce permissions, approval and identity in the executor.
2. **Retrying side effects blindly.** A timeout does not say whether the action happened; use
   idempotency keys or report "outcome unknown".
3. **Trusting text from tools and servers.** Descriptions and results are data written by someone else;
   lint, pin and never follow instructions found in them.

---

[🏠 Module 07](README.md) · [Quiz](quiz.md) · [Lab](lab/README.md)
