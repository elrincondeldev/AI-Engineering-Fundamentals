"""Reflexion: try, evaluate, write a lesson in words, and try again with that lesson in memory.

The evaluator here never sees the gold answer. It only uses signals available at
run time (the trajectory, the pages read, the answer text). That is the realistic
setting, and it is why Reflexion can only fix failures its evaluator can detect.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .agents import Result, run_react
from .credibility import text_spam_signals
from .model import Context, ScriptedModel, facts_for, parse_page
from .providers import FakeProvider, LLMProvider, Message
from .sanitize import looks_like_injection
from .skills import analyze, same_value
from .tools import WebTools


def self_evaluate(result: Result) -> list[str]:
    """Problems the agent can see in its own trajectory, as tagged feedback lines."""
    problems = []
    pages = [p for s in result.steps if s.tool == "open_page" and (p := parse_page(s.observation, 0, 0))]
    cited = set(result.sources)
    body = result.answer_text.split("Sources:")[0].lower()
    injected = [
        p for p in pages if any(looks_like_injection(x) for x in re.split(r"(?<=[.!?])\s+|\n", p.text))
    ]
    foreign = {d for d in re.findall(r"\b([a-z0-9-]+\.example)\b", body)} - {u.split("/")[2] for u in cited}
    if injected and foreign:
        problems.append(
            f"[injection] {injected[0].url} addressed AI assistants; the answer mentions {sorted(foreign)}."
        )
    for p in pages:
        if p.url in cited and text_spam_signals("", p.text).is_spam:
            problems.append(f"[spam] the answer cites {p.url}, which shows spam signals.")
    plan = analyze(result.question)
    if plan.wants_recent:
        ctx = Context(result.question, opened=[replace(p, order=i) for i, p in enumerate(pages)])
        for slot in plan.slots:
            facts = facts_for(slot, ctx)
            used = [f for f in facts if f.url in cited]
            newer = [
                f for f in facts if used and f.published > used[0].published and not same_value(f, used[0])
            ]
            if newer:
                problems.append(
                    f"[stale] {used[0].url} ({used[0].published}) is older than {newer[0].url} "
                    f"({newer[0].published}) and they disagree on the {slot.describe()}."
                )
    if result.short.lower().startswith("unknown"):
        problems.append("[unknown] the answer is unknown.")
    return problems


def run_reflexion(
    question: str, tools: WebTools, provider: LLMProvider | None = None, max_trials: int = 3
) -> Result:
    """Up to ``max_trials`` ReAct attempts; between attempts, reflect and remember."""
    model = ScriptedModel()
    reflector = provider or FakeProvider(responder=model.reflect)
    memory: list[str] = []
    total = Result("reflexion", question, "")
    last: Result | None = None
    for trial in range(1, max_trials + 1):
        last = run_react(question, tools, provider, reflections=tuple(memory))
        total.usage.merge(last.usage)
        total.steps += last.steps
        total.prompt_sizes += last.prompt_sizes
        for name in ("tool_calls", "tool_errors", "redundant_calls", "injections_removed"):
            setattr(total, name, getattr(total, name) + getattr(last, name))
        problems = self_evaluate(last)
        problems = [
            p for p in problems if not (p.startswith("[unknown]") and any("broader" in m for m in memory))
        ]
        total.notes.append(f"trial {trial}: {last.short!r} | problems: {problems or 'none'}")
        if not problems or trial == max_trials:
            break
        out = reflector.complete(
            [
                Message("system", "Write one lesson for your next attempt."),
                Message("user", "Feedback:\n" + "\n".join(problems)),
            ]
        )
        total.usage.add(out)
        total.prompt_sizes.append(out.prompt_tokens)
        if out.text not in memory:
            memory.append(out.text)
        total.notes.append(f"reflection: {out.text}")
    assert last is not None
    total.answer_text = last.answer_text
    total.stopped = last.stopped
    return total
