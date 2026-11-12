"""A supervisor with two specialists, talking through A2A-style tasks, messages and artifacts.

This is an in-process simulation of the *ideas* in the Agent2Agent (A2A) protocol:
agents publish an agent card that says what they can do, a client sends a message
that creates a task, the task moves through states, and results come back as
artifacts. Real A2A runs over HTTP with JSON-RPC 2.0; check the specification for
the exact method names and fields. Nothing here opens a socket.

Agents:
- supervisor: splits the question into facts, delegates, and writes the final answer.
- researcher: a ReAct agent that finds one fact (skill ``web_research``).
- fact_checker: looks for a second source and may overrule a claim (skill ``fact_check``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from itertools import count

from .agents import Result, run_react, system_prompt
from .credibility import text_spam_signals
from .model import Context, Opened, facts_for, parse_page
from .providers import FakeProvider, LLMProvider, Message
from .skills import Fact, Slot, analyze, compose, resolve, same_value
from .tools import WebTools

STATES = ("submitted", "working", "input-required", "completed", "failed")


@dataclass(frozen=True)
class AgentCard:
    """What an agent advertises so others can discover it."""

    name: str
    description: str
    skills: tuple[str, ...]
    endpoint: str  # where a real deployment would accept requests (unused offline)


@dataclass
class Part:
    kind: str  # "text" or "data"
    content: str | dict


@dataclass
class A2AMessage:
    role: str  # "user" (the caller) or "agent"
    parts: list[Part]


@dataclass
class Artifact:
    name: str
    parts: list[Part]


@dataclass
class Task:
    """A unit of work with a lifecycle. ``history`` keeps every state it went through."""

    id: str
    agent: str
    state: str = "submitted"
    messages: list[A2AMessage] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    history: list[str] = field(default_factory=lambda: ["submitted"])

    def move(self, state: str) -> None:
        assert state in STATES, state
        self.state = state
        self.history.append(state)


def jsonrpc_envelope(method: str, params: dict, request_id: int) -> dict:
    """The JSON-RPC 2.0 shape a networked deployment would send (method names: see the A2A spec)."""
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


Handler = Callable[[Task], None]


class Directory:
    """An in-process registry: discover agents by skill and send them tasks."""

    def __init__(self) -> None:
        self.cards: dict[str, AgentCard] = {}
        self.handlers: dict[str, Handler] = {}
        self.wire: list[dict] = []  # every envelope "sent", for inspection and counting
        self._ids = count(1)

    def register(self, card: AgentCard, handler: Handler) -> None:
        self.cards[card.name] = card
        self.handlers[card.name] = handler

    def discover(self, skill: str) -> AgentCard:
        for card in self.cards.values():
            if skill in card.skills:
                return card
        raise LookupError(f"no agent offers {skill!r}")

    def send(self, card: AgentCard, message: A2AMessage) -> Task:
        n = next(self._ids)
        task = Task(id=f"task-{n}", agent=card.name, messages=[message])
        self.wire.append(
            jsonrpc_envelope("send_message", {"agent": card.name, "message": asdict(message)}, n)
        )
        task.move("working")
        self.handlers[card.name](task)
        self.wire.append({"jsonrpc": "2.0", "id": n, "result": {"task": task.id, "state": task.state}})
        return task


@dataclass
class MultiAgentSystem:
    """Supervisor + researcher + fact-checker over shared tools, with one shared bill."""

    tools: WebTools
    provider: LLMProvider | None = None
    critic: bool = True
    structured_handoff: bool = True
    result: Result = field(default_factory=lambda: Result("multi", "", ""))

    def __post_init__(self) -> None:
        self.directory = Directory()
        self.directory.register(
            AgentCard(
                "researcher",
                "Finds one fact on the offline web and cites it.",
                ("web_research",),
                "local://researcher",
            ),
            self._research,
        )
        self.directory.register(
            AgentCard(
                "fact_checker",
                "Checks a claimed fact against a second source.",
                ("fact_check",),
                "local://fact_checker",
            ),
            self._check,
        )
        self._llm = self.provider or FakeProvider(responder=_scripted_agents)

    # --------------------------------------------------------------- specialists
    def _research(self, task: Task) -> None:
        """Researcher: a ReAct run for one group of facts that share a search query."""
        directives = str(task.messages[0].parts[0].content)
        sub = run_react(
            f"Find these facts. {directives}", self.tools, self.provider, max_steps=6, reset_tools=False
        )
        self._bill(sub)
        pages = [p for s in sub.steps if s.tool == "open_page" and (p := parse_page(s.observation, 0, 0))]
        plan = analyze(directives)
        ctx = Context(plan.question, opened=pages)
        found = {}
        for slot in plan.slots:
            fact = resolve(facts_for(slot, ctx), self.tools.guards.credibility, self.tools.guards.recency)
            if fact:
                excerpt = next(p.text[:400] for p in pages if p.url == fact.url)
                found[slot.key] = {
                    "value": fact.value,
                    "url": fact.url,
                    "published": fact.published,
                    "tier": fact.tier,
                    "excerpt": excerpt,
                }
        task.artifacts.append(Artifact("facts", [Part("data", found), Part("text", sub.answer_text)]))
        task.move("completed")

    def _check(self, task: Task) -> None:
        """Fact-checker: read up to two other results for the same query and judge each claim."""
        claims = task.messages[0].parts[0].content
        assert isinstance(claims, list)
        query = claims[0]["slot"]["query"]
        claimed = {c["url"] for c in claims}
        urls = [u for u in re.findall(r"url: (\S+)", self.tools.search_web(query)) if u not in claimed][:2]
        pages = [p for u in urls if (p := parse_page(self.tools.open_page(u), 0, 0))]
        payload = {
            "claims": claims,
            "others": [asdict(p) for p in pages],
            "prefer_recent": self.tools.guards.recency,
        }
        out = self._llm.complete(
            [
                Message("system", "You are a sceptical fact-checker."),
                Message("user", "CHECK " + json.dumps(payload)),
            ]
        )
        self._account(out)
        task.artifacts.append(Artifact("verdicts", [Part("data", json.loads(out.text))]))
        task.move("completed")

    # --------------------------------------------------------------- supervisor
    def run(self, question: str) -> Result:
        self.tools.reset()
        self.result = Result("multi", question, "")
        out = self._llm.complete(
            [
                Message("system", system_prompt(self.tools.guards, "planner")),
                Message("user", "PLAN " + question),
            ]
        )
        self._account(out)
        plan = analyze(question)
        researcher = self.directory.discover("web_research")
        checker = self.directory.discover("fact_check")
        groups: dict[str, list[Slot]] = {}
        for slot in plan.slots:
            groups.setdefault(slot.query, []).append(slot)
        chosen: dict[str, Fact | None] = {s.key: None for s in plan.slots}
        notes: list[str] = []
        for slots in groups.values():
            directives = " ".join(s.directive() for s in slots)
            task = self.directory.send(researcher, A2AMessage("user", [Part("text", directives)]))
            found = task.artifacts[0].parts[0].content
            assert isinstance(found, dict)
            if not self.structured_handoff:
                notes.append(str(task.artifacts[0].parts[1].content))
            for slot in slots:
                if slot.key in found:
                    chosen[slot.key] = _fact(slot, found[slot.key])
            claims = [{"slot": asdict(s), **found[s.key]} for s in slots if s.key in found]
            if self.critic and claims:
                verdicts = self.directory.send(checker, A2AMessage("user", [Part("data", claims)]))
                for key, verdict in verdicts.artifacts[0].parts[0].content.items():  # type: ignore[union-attr]
                    self.result.notes.append(f"{key}: {verdict['verdict']}")
                    if verdict["verdict"] == "overruled":
                        slot = next(s for s in slots if s.key == key)
                        chosen[key] = _fact(slot, verdict)
        answer = compose(plan, chosen)
        payload = json.dumps({"answer": answer.render(), "researcher_notes": notes})
        out = self._llm.complete(
            [Message("system", "Write the final answer from the facts."), Message("user", "WRITE " + payload)]
        )
        self._account(out)
        self.result.answer_text = out.text
        log = self.tools.log
        self.result.tool_calls, self.result.tool_errors = len(log.calls), log.errors
        self.result.redundant_calls, self.result.injections_removed = log.redundant, log.injections_removed
        self.result.notes.append(f"messages on the wire: {len(self.directory.wire)}")
        return self.result

    def _account(self, out) -> None:
        self.result.usage.add(out)
        self.result.prompt_sizes.append(out.prompt_tokens)

    def _bill(self, sub: Result) -> None:
        self.result.usage.merge(sub.usage)
        self.result.prompt_sizes += sub.prompt_sizes
        self.result.steps += sub.steps


def _fact(slot: Slot, data: dict) -> Fact:
    items = tuple(data["value"].split(", ")) if slot.attribute == "countries" else ()
    return Fact(
        slot.key, data["value"], data["url"], data.get("published", ""), data.get("tier", ""), "", items=items
    )


def _scripted_agents(messages: list[Message]) -> str:
    """Responder for the supervisor's own calls and the fact-checker's verdicts."""
    text = messages[-1].content
    if text.startswith("PLAN "):
        plan = analyze(text[5:])
        return "\n".join(f"- delegate: {s.directive()}" for s in plan.slots)
    if text.startswith("WRITE "):
        data = json.loads(text[6:])
        answer = data["answer"]
        for note in data["researcher_notes"]:  # a gullible writer pastes what the researchers said
            for extra in re.findall(r"Note: [^\n]+", note):
                answer = answer.replace("\nSources:", f" {extra}\nSources:", 1)
        return answer
    data = json.loads(text[len("CHECK ") :])
    others = [
        Opened(o["url"], o["published"], o["tier"], o["text"], i + 1, 0) for i, o in enumerate(data["others"])
    ]
    clean = [o for o in others if not text_spam_signals("", o.text).is_spam]
    verdicts = {}
    for claim in data["claims"]:
        slot = Slot(**claim["slot"])
        mine: Fact | None = _fact(slot, claim)
        if text_spam_signals("", claim["excerpt"]).is_spam:
            mine = None  # a sceptic does not trust pages that look like spam
        alternatives = facts_for(slot, Context(slot.directive(), opened=clean))
        pool = ([mine] if mine else []) + alternatives
        best = resolve(pool, True, data["prefer_recent"] or "latest" in slot.query)
        if best is None:
            verdicts[slot.key] = {"verdict": "unverified"}
        elif mine and same_value(best, mine):
            agree = any(same_value(f, mine) for f in alternatives)
            verdicts[slot.key] = {"verdict": "confirmed" if agree else "unverified"}
        else:
            verdicts[slot.key] = {
                "verdict": "overruled",
                "value": best.value,
                "url": best.url,
                "published": best.published,
            }
    return json.dumps(verdicts)


def run_multi(
    question: str,
    tools: WebTools,
    provider: LLMProvider | None = None,
    critic: bool = True,
    structured_handoff: bool = True,
) -> Result:
    return MultiAgentSystem(tools, provider, critic, structured_handoff).run(question)
