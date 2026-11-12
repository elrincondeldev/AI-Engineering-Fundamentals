"""Agent registry, run recording and offline replay.

Recording every run as JSON lines means you can re-grade, compare and debug later
without paying for a single model call again (``lab replay``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .agents import Result, Step, Usage, run_react, run_rewoo
from .evaluate import Graded, grade, load_tasks
from .multi import run_multi
from .providers import LLMProvider, get_provider
from .reflexion import run_reflexion
from .search import Guards
from .tools import WebTools
from .tree import run_tree

Runner = Callable[..., Result]

AGENTS: dict[str, Runner] = {
    "react": lambda q, t, p, max_steps=12, **_: run_react(q, t, p, max_steps=max_steps),
    "rewoo": lambda q, t, p, **_: run_rewoo(q, t, p),
    "plan": lambda q, t, p, **_: run_rewoo(q, t, p, replan=True),
    "reflexion": lambda q, t, p, **_: run_reflexion(q, t, p),
    # The value model and the fact-checker expect structured replies, so these stay scripted.
    "tree": lambda q, t, p, branching=3, beam=2, **_: run_tree(
        q, t, strategy="beam", branching=branching, beam=beam
    ),
    "mcts": lambda q, t, p, branching=3, **_: run_tree(q, t, strategy="mcts", branching=branching),
    "multi": lambda q, t, p, critic=True, structured_handoff=True, **_: run_multi(
        q, t, critic=critic, structured_handoff=structured_handoff
    ),
}


def live_provider() -> LLMProvider | None:
    """None means 'use the scripted model'. Any other LLM_PROVIDER drives the same loops."""
    provider = get_provider()
    return None if provider.name == "fake" else provider


def run_agent(
    agent: str, question: str, guards: Guards = Guards(), tools: WebTools | None = None, **options: object
) -> Result:
    """Run one agent on one question. ``options`` go to the agent (max_steps, branching, critic...)."""
    if agent not in AGENTS:
        raise SystemExit(f"Unknown agent {agent!r}. Choose one of: {', '.join(AGENTS)}")
    return AGENTS[agent](question, tools or WebTools(guards), live_provider(), **options)


def run_eval(
    agent: str, guards: Guards = Guards(), tasks: list[dict] | None = None, **options: object
) -> list[Graded]:
    """Run one agent on every task (one shared tool set) and grade each answer."""
    tools = WebTools(guards)
    return [
        grade(run_agent(agent, t["question"], guards, tools, **options), t) for t in tasks or load_tasks()
    ]


def result_to_record(task_id: str, guards: Guards, r: Result) -> dict:
    """A JSON-safe record; observations are stored as short hashes to keep files small."""
    record = asdict(r)
    record["task_id"] = task_id
    record["guards"] = guards.label()
    for step in record["steps"]:
        step["observation"] = hashlib.sha1(step["observation"].encode()).hexdigest()[:12]
    return record


def record_from_dict(d: dict) -> Result:
    d = dict(d)
    d.pop("task_id", None)
    d.pop("guards", None)
    d["usage"] = Usage(**d["usage"])
    d["steps"] = [Step(**s) for s in d["steps"]]
    return Result(**d)


def save_run(path: Path, graded: list[Graded], guards: Guards) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for g in graded:
            fh.write(json.dumps(result_to_record(g.task["id"], guards, g.result), ensure_ascii=False) + "\n")


def replay(path: Path) -> list[Graded]:
    """Re-grade a saved run. No model, no tools: only the recorded answers."""
    tasks = {t["id"]: t for t in load_tasks()}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [grade(record_from_dict(row), tasks[row["task_id"]]) for row in rows]
