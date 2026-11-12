"""Single-agent control loops: ReAct, ReWOO and plan-and-execute.

The loops are model-agnostic: they send messages to an ``LLMProvider`` and parse
text back. With the default ``FakeProvider`` the scripted model decides; set
``LLM_PROVIDER=ollama`` or ``openai`` to let a real model drive the same loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .model import RULE_CREDIBLE, RULE_DATA, RULE_RECENT, RULE_VERIFY, ScriptedModel
from .providers import Completion, FakeProvider, LLMProvider, Message
from .search import Guards
from .tools import WebTools

MAX_STEPS = 12

TOOLS_HELP = """Tools:
- search_web[query]: search the offline web snapshot; returns up to 5 results.
- open_page[url]: read one page from the snapshot."""

REACT_FORMAT = """Work in steps. Each turn, write exactly:
Thought: what you know and what you need next
Action: search_web[...] or open_page[...]
Then wait for the Observation. When you can answer, write instead:
Thought: ...
Final Answer: the short answer (or "unknown" if the pages do not say)
Explanation: one or two sentences that cite sources as [1], [2]
Sources: [1] url [2] url"""


def system_prompt(guards: Guards, role: str = "react") -> str:
    """The instructions the model sees. Guard rules appear only when the guard is on."""
    rules = [RULE_DATA, RULE_VERIFY]
    if guards.credibility:
        rules.insert(0, RULE_CREDIBLE)
    if guards.recency:
        rules.insert(0, RULE_RECENT)
    head = "You are Hearthwise's Ask-the-Web research assistant. Use only the tools; never guess."
    body = (
        REACT_FORMAT
        if role == "react"
        else (
            "Write a plan before seeing any evidence. Each step is 'Plan: ...' followed by\n"
            "'#E<n> = tool[argument]'. Refer to the k-th result of a search as #E<n>.<k>."
            if role == "planner"
            else "Answer from the evidence only, in the Final Answer format."
        )
    )
    return "\n".join([head, TOOLS_HELP, "Rules:", *[f"- {r}" for r in rules], body])


def user_prompt(question: str, reflections: tuple[str, ...] = ()) -> str:
    lines = [f"Question: {question}"]
    if reflections:
        lines.append("Lessons from your earlier attempts:")
        lines += [f"- Reflection: {r}" for r in reflections]
    return "\n".join(lines)


@dataclass
class Usage:
    """Model calls and tokens for one run (tools are free; model calls are not)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, c: Completion) -> None:
        self.calls += 1
        self.prompt_tokens += c.prompt_tokens
        self.completion_tokens += c.completion_tokens

    def merge(self, other: Usage) -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Step:
    thought: str
    tool: str
    arg: str
    observation: str


@dataclass
class Result:
    """What one agent run produced: the answer text, the trajectory and the bill."""

    agent: str
    question: str
    answer_text: str
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    tool_calls: int = 0
    tool_errors: int = 0
    redundant_calls: int = 0
    injections_removed: int = 0
    stopped: str = "answer"  # answer | budget
    notes: list[str] = field(default_factory=list)
    prompt_sizes: list[int] = field(default_factory=list)  # prompt tokens of each model call

    @property
    def short(self) -> str:
        m = re.search(r"^Final Answer:\s*(.+)$", self.answer_text, re.M)
        return m.group(1).strip() if m else self.answer_text.strip()

    @property
    def sources(self) -> list[str]:
        m = re.search(r"^Sources:\s*(.+)$", self.answer_text, re.M)
        return re.findall(r"\[\d+\]\s+(\S+)", m.group(1)) if m else []


def _finish(result: Result, tools: WebTools) -> Result:
    log = tools.log
    result.tool_calls = len(log.calls)
    result.tool_errors = log.errors
    result.redundant_calls = log.redundant
    result.injections_removed = log.injections_removed
    return result


def _call(provider: LLMProvider, messages: list[Message], usage: Usage, result: Result, **kw) -> str:
    out = provider.complete(messages, temperature=0.0, max_tokens=600, seed=0, **kw)
    usage.add(out)
    result.prompt_sizes.append(out.prompt_tokens)
    return out.text


def parse_action(text: str) -> tuple[str, str]:
    """Return ('final', text), (tool, argument) or ('invalid', text)."""
    if re.search(r"^Final Answer:", text, re.M):
        return "final", text[text.index("Final Answer:") :]
    m = re.search(r"^Action:\s*(\w+)\[(.*)\]\s*$", text, re.M)
    if m:
        return m.group(1), m.group(2).strip()
    return "invalid", text


def run_react(
    question: str,
    tools: WebTools,
    provider: LLMProvider | None = None,
    max_steps: int = MAX_STEPS,
    reflections: tuple[str, ...] = (),
    reset_tools: bool = True,
) -> Result:
    """Thought -> Action -> Observation until a final answer or the step budget runs out."""
    provider = provider or FakeProvider(responder=ScriptedModel().react_step)
    if reset_tools:
        tools.reset()
    result = Result("react", question, "")
    messages = [
        Message("system", system_prompt(tools.guards)),
        Message("user", user_prompt(question, reflections)),
    ]
    for _ in range(max_steps):
        text = _call(provider, messages, result.usage, result, stop=["\nObservation:"])
        kind, arg = parse_action(text)
        if kind == "final":
            result.answer_text = arg
            return _finish(result, tools)
        if kind == "invalid":
            obs = "ERROR: could not parse an Action. Use the exact format."
            tools.log.errors += 1
        else:
            obs = tools.call(kind, arg)
        thought = text.split("\nAction:")[0].removeprefix("Thought:").strip()
        result.steps.append(Step(thought, kind, arg, obs))
        messages += [Message("assistant", text), Message("user", f"Observation: {obs}")]
    messages.append(Message("user", "You are out of steps. Give your final answer now."))
    text = _call(provider, messages, result.usage, result)
    result.answer_text = (
        parse_action(text)[1] if parse_action(text)[0] == "final" else "Final Answer: unknown"
    )
    result.stopped = "budget"
    return _finish(result, tools)


def _execute_plan(plan_text: str, tools: WebTools, result: Result, evidence: dict[int, str]) -> list[str]:
    """ReWOO workers: run each '#En = tool[arg]' line, substituting '#Ek.r' with result r of #Ek."""
    blocks = []
    for m in re.finditer(r"^#E(\d+) = (\w+)\[(.*)\]\s*$", plan_text, re.M):
        n, tool, arg = int(m.group(1)), m.group(2), m.group(3)
        ref = re.fullmatch(r"#E(\d+)\.(\d+)", arg)
        if ref:
            urls = re.findall(r"url: (\S+)", evidence.get(int(ref.group(1)), ""))
            k = int(ref.group(2))
            if k > len(urls):
                tools.log.errors += 1
                evidence[n] = f"ERROR: #E{ref.group(1)} has no result {k}."
                blocks.append(f"#E{n} = {tool}[{arg}]\n{evidence[n]}")
                continue
            arg = urls[k - 1]
        evidence[n] = tools.call(tool, arg)
        result.steps.append(Step("(planned)", tool, arg, evidence[n]))
        blocks.append(f"#E{n} = {tool}[{arg}]\n{evidence[n]}")
    return blocks


def run_rewoo(
    question: str, tools: WebTools, provider: LLMProvider | None = None, replan: bool = False
) -> Result:
    """Plan once, let workers run every tool call, then solve in one more model call.

    With ``replan=True`` it becomes a small plan-and-execute agent: if the solver is
    missing facts, a re-planner writes extra steps once, and the solver runs again.
    """
    model = ScriptedModel()
    planner = provider or FakeProvider(responder=model.rewoo_plan)
    solver = provider or FakeProvider(responder=model.rewoo_solve)
    replanner = provider or FakeProvider(responder=model.replan)
    tools.reset()
    result = Result("plan_execute" if replan else "rewoo", question, "")
    user = Message("user", user_prompt(question))
    plan_text = _call(
        planner, [Message("system", system_prompt(tools.guards, "planner")), user], result.usage, result
    )
    evidence: dict[int, str] = {}
    blocks = _execute_plan(plan_text, tools, result, evidence)
    solve_sys = Message("system", system_prompt(tools.guards, "solver"))
    text = _call(
        solver, [solve_sys, user, Message("user", "Evidence:\n" + "\n\n".join(blocks))], result.usage, result
    )
    if replan and "Missing:" in text:
        more = _call(
            replanner,
            [
                Message("system", system_prompt(tools.guards, "planner")),
                user,
                Message("user", "Evidence:\n" + "\n\n".join(blocks)),
            ],
            result.usage,
            result,
        )
        result.notes.append(more)
        blocks += _execute_plan(more, tools, result, evidence)
        text = _call(
            solver,
            [solve_sys, user, Message("user", "Evidence:\n" + "\n\n".join(blocks))],
            result.usage,
            result,
        )
    result.answer_text = parse_action(text)[1]
    result.notes.insert(0, plan_text)
    return _finish(result, tools)
