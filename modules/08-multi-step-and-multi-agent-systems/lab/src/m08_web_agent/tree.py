"""Tree search over evidence choices: beam search and a small Monte Carlo tree search.

A greedy agent (ReAct) commits to the first page that answers a sub-question. A tree
search keeps several alternatives alive and asks a *value function* which partial
answer looks most trustworthy. Here a decision level is "which search result to use
for this group of facts"; a node is the evidence chosen so far.

The value function is a model call (as in LATS, where the LLM scores states). Our
scripted value model reads each chosen fact's source and scores it from visible
signals: spam signals in the text, the source tier when shown, agreement with the
other candidates it has seen, and recency when the question asks for the latest value.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .agents import Result, Step, Usage
from .credibility import TIER_SCORE, recency_score, text_spam_signals
from .model import Context, Opened, injected_demands, parse_page
from .providers import FakeProvider, LLMProvider, Message
from .skills import Answer, Fact, Plan, Slot, analyze, compose, extract, same_value
from .tools import WebTools


@dataclass
class Level:
    """One decision: the slots that share a query and the pages we may take them from."""

    query: str
    slots: list[Slot]
    pages: list[Opened] = field(default_factory=list)


@dataclass
class Node:
    """The evidence chosen for the first ``depth`` levels."""

    picks: tuple[Opened, ...] = ()
    value: float = 0.0
    visits: int = 0
    total: float = 0.0
    children: list[Node] = field(default_factory=list)
    expanded: int = 0  # how many candidate pages have become children


def value_model(messages: Sequence[Message]) -> str:
    """Scripted value function: reads the evidence JSON and returns 'Score: x'."""
    data = json.loads(messages[-1].content.split("Evidence:", 1)[1])
    if not data["facts"]:
        return "Score: 0.000"
    scores = []
    for f in data["facts"]:
        if f["value"] is None:
            scores.append(0.0)
            continue
        if text_spam_signals("", f["excerpt"]).is_spam:
            scores.append(0.05)
            continue
        s = TIER_SCORE.get(f["tier"], 0.5)
        s += 0.25 * min(f["agreeing_candidates"], 2)
        if data["wants_recent"]:
            s += 0.3 * recency_score(f["published"])
        scores.append(min(s, 1.0))
    return f"Score: {sum(scores) / len(scores):.3f}"


class TreeSearch:
    """Beam search (default) or MCTS over which page supplies each group of facts."""

    def __init__(
        self,
        tools: WebTools,
        provider: LLMProvider | None = None,
        branching: int = 3,
        beam: int = 2,
        iterations: int = 12,
        seed: int = 0,
    ) -> None:
        self.tools = tools
        self.valuer = provider or FakeProvider(responder=value_model)
        self.branching, self.beam, self.iterations = branching, beam, iterations
        self.rng = random.Random(seed)
        self.usage = Usage()
        self.prompt_sizes: list[int] = []
        self.nodes_scored = 0
        self.steps: list[Step] = []
        self.trace: list[str] = []

    # ---------------------------------------------------------------- set-up
    def levels(self, plan: Plan) -> list[Level]:
        groups: dict[str, Level] = {}
        for slot in plan.slots:
            groups.setdefault(slot.query, Level(slot.query, [])).slots.append(slot)
        cache: dict[str, Opened] = {}
        for level in groups.values():
            obs = self.tools.search_web(level.query)
            self.steps.append(Step("expand: one search per fact group", "search_web", level.query, obs))
            for url in re.findall(r"url: (\S+)", obs)[: self.branching]:
                if url not in cache:
                    seen = self.tools.open_page(url)
                    self.steps.append(Step("expand: read a candidate", "open_page", url, seen))
                    page = parse_page(seen, len(cache), 0)
                    if page is None:
                        continue
                    cache[url] = page
                level.pages.append(cache[url])
        return list(groups.values())

    def facts(self, levels: list[Level], node: Node) -> list[tuple[Slot, Fact | None, Level]]:
        out = []
        for level, page in zip(levels, node.picks):
            for slot in level.slots:
                out.append(
                    (slot, extract(slot, page.text, page.url, page.published, page.tier, page.order), level)
                )
        return out

    def score(self, plan: Plan, levels: list[Level], node: Node) -> float:
        """Ask the value model how trustworthy this partial evidence looks."""
        items = []
        for slot, fact, level in self.facts(levels, node):
            others = [extract(slot, p.text, p.url, p.published, p.tier) for p in level.pages]
            agreeing = sum(1 for o in others if fact and o and o.url != fact.url and same_value(o, fact))
            page = next(p for p in level.pages if fact is None or p.url == fact.url)
            items.append(
                {
                    "slot": slot.describe(),
                    "value": fact.value if fact else None,
                    "url": fact.url if fact else None,
                    "tier": page.tier,
                    "published": page.published,
                    "agreeing_candidates": agreeing,
                    "excerpt": page.text[:400],
                }
            )
        prompt = json.dumps({"question": plan.question, "wants_recent": plan.wants_recent, "facts": items})
        out = self.valuer.complete(
            [
                Message("system", "Rate how well this evidence answers the question, 0 to 1."),
                Message("user", "Evidence:" + prompt),
            ]
        )
        self.usage.add(out)
        self.prompt_sizes.append(out.prompt_tokens)
        self.nodes_scored += 1
        m = re.search(r"Score:\s*([\d.]+)", out.text)
        value = float(m.group(1)) if m else 0.0
        last = items[-1] if items else {"url": None, "value": None}
        where = (last["url"] or node.picks[-1].url).split("//")[1]
        self.trace.append(f"depth {len(node.picks)} · {where} -> {last['value']} · score {value:.3f}")
        return value

    # ---------------------------------------------------------------- beam search
    def beam_search(self, plan: Plan, levels: list[Level]) -> Node:
        frontier = [Node()]
        for level in levels:
            children = []
            for node in frontier:
                for page in level.pages:
                    child = Node(node.picks + (page,))
                    child.value = self.score(plan, levels, child)
                    children.append(child)
            children.sort(key=lambda n: (-n.value, [p.order for p in n.picks]))
            frontier = children[: self.beam] or frontier
        return frontier[0]

    # ---------------------------------------------------------------- MCTS
    def mcts(self, plan: Plan, levels: list[Level], c: float = 1.0) -> Node:
        root = Node()
        best: Node | None = None
        for _ in range(self.iterations):
            path, node = [root], root
            while node.children and node.expanded >= len(levels[len(node.picks)].pages):  # selection (UCT)
                parent = node
                node = max(
                    node.children,
                    key=lambda ch: (
                        ch.total / ch.visits + c * math.sqrt(math.log(parent.visits + 1) / ch.visits),
                        self.rng.random(),
                    ),
                )
                path.append(node)
                if len(node.picks) == len(levels):
                    break
            depth = len(node.picks)
            if depth < len(levels) and node.expanded < len(levels[depth].pages):  # expansion
                child = Node(node.picks + (levels[depth].pages[node.expanded],))
                node.expanded += 1
                node.children.append(child)
                child.value = self.score(plan, levels, child)  # evaluation by the value model
                path.append(child)
                node = child
            reward = node.value
            for n in path:  # backpropagation
                n.visits += 1
                n.total += reward
            if len(node.picks) == len(levels) and (best is None or node.value > best.value):
                best = node
        if best is None:  # budget too small to reach a leaf: finish greedily
            node = root
            while len(node.picks) < len(levels):
                node = (
                    max(node.children, key=lambda ch: ch.value)
                    if node.children
                    else Node(node.picks + (levels[len(node.picks)].pages[0],))
                )
            best = node
        return best


def run_tree(
    question: str, tools: WebTools, provider: LLMProvider | None = None, strategy: str = "beam", **kwargs
) -> Result:
    """Plan the facts, search once per fact group, then pick evidence with beam search or MCTS."""
    tools.reset()
    search = TreeSearch(tools, provider, **kwargs)
    plan = analyze(question)
    levels = search.levels(plan)
    result = Result(f"tree-{strategy}", question, "")
    if not all(level.pages for level in levels):
        answer = compose(plan, {s.key: None for s in plan.slots})
    else:
        node = search.beam_search(plan, levels) if strategy == "beam" else search.mcts(plan, levels)
        chosen = {slot.key: fact for slot, fact, _ in search.facts(levels, node)}
        answer = compose(plan, chosen)
        for demand in injected_demands(Context(question, opened=list(node.picks))):  # same gullible writer
            answer = Answer(
                answer.short, f"{answer.explanation} Note: {demand}.", answer.sources, answer.support
            )
        result.notes.append(f"chosen: {[p.url for p in node.picks]} value={node.value:.3f}")
    result.answer_text = answer.render()
    result.steps = search.steps
    result.notes = search.trace + result.notes
    result.usage = search.usage
    result.prompt_sizes = search.prompt_sizes
    result.notes.append(f"nodes scored: {search.nodes_scored}")
    log = tools.log
    result.tool_calls, result.tool_errors, result.redundant_calls = len(log.calls), log.errors, log.redundant
    result.injections_removed = log.injections_removed
    return result
