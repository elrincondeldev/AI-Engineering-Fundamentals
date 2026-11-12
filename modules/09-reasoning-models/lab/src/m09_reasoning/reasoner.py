"""The client every algorithm uses: builds prompts, calls the provider, parses answers, counts tokens."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from . import prompts
from .problems import Problem
from .providers import Completion, FakeProvider, LLMProvider, Message, count_tokens, get_provider
from .puzzle import Item, exact, find_move, parse_move_line
from .simulator import SimConfig, SimulatedReasoner

_STEP_RE = re.compile(r"^\s*Step (\d+):\s*(.*?)\s*$")
_ANSWER_RE = re.compile(r"^\s*Answer:\s*(.+?)\s*$", re.M)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


@dataclass
class Usage:
    """Token bill. ``thinking_tokens`` is the part of ``completion_tokens`` written inside <think>."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0

    def add(self, completion: Completion, thinking: int = 0) -> None:
        self.calls += 1
        self.prompt_tokens += completion.prompt_tokens
        self.completion_tokens += completion.completion_tokens
        self.thinking_tokens += thinking

    def snapshot(self) -> Usage:
        return Usage(self.calls, self.prompt_tokens, self.completion_tokens, self.thinking_tokens)

    def minus(self, earlier: Usage) -> Usage:
        return Usage(
            self.calls - earlier.calls,
            self.prompt_tokens - earlier.prompt_tokens,
            self.completion_tokens - earlier.completion_tokens,
            self.thinking_tokens - earlier.thinking_tokens,
        )


@dataclass(frozen=True)
class Attempt:
    """One parsed solution."""

    text: str
    steps: tuple[str, ...]
    answer: str | None
    completion_tokens: int
    thinking_tokens: int = 0


def parse_solution(text: str) -> tuple[tuple[str, ...], str | None, int]:
    """Return (final steps, answer, thinking tokens). A step rewritten later replaces the earlier one."""
    think = _THINK_RE.search(text)
    thinking = count_tokens(think.group(1)) if think else 0
    visible = text[think.end() :] if think else text
    steps: dict[int, str] = {}
    for line in visible.splitlines():
        m = _STEP_RE.match(line)
        if m:
            steps[int(m.group(1))] = m.group(2)
    answers = _ANSWER_RE.findall(visible)
    return tuple(steps[k] for k in sorted(steps)), (answers[-1] if answers else None), thinking


@dataclass
class Reasoner:
    """Talks to any :class:`LLMProvider`. With the default fake provider it talks to the simulator."""

    provider: LLMProvider
    temperature: float = 0.8
    usage: Usage = field(default_factory=Usage)

    def _ask(self, system: str, user: str) -> Completion:
        out = self.provider.complete(
            [Message("system", system), Message("user", user)],
            temperature=self.temperature,
            max_tokens=100_000,
        )
        return out

    def _attempt(self, out: Completion) -> Attempt:
        steps, answer, thinking = parse_solution(out.text)
        self.usage.add(out, thinking)
        return Attempt(out.text, steps, answer, out.completion_tokens, thinking)

    # -- whole solutions ------------------------------------------------------

    def solve(self, problem: Problem, mode: str = "cot", effort: str = "medium") -> Attempt:
        """mode: 'direct' (answer only), 'cot' (step by step) or 'think' (hidden reasoning + effort)."""
        system = {"direct": prompts.SOLVE_DIRECT, "cot": prompts.SOLVE_COT}.get(mode)
        if mode == "think":
            system = prompts.think_prompt(effort)
        if system is None:
            raise ValueError(f"unknown mode {mode!r}")
        return self._attempt(self._ask(system, prompts.problem_message(problem.question)))

    def solve_with_hint(self, problem: Problem) -> Attempt:
        """STaR rationalisation: show the answer and ask for a rationale that reaches it."""
        return self._attempt(
            self._ask(prompts.SOLVE_COT, prompts.hint_message(problem.question, problem.answer))
        )

    def revise(self, problem: Problem, attempt: Attempt, feedback: str) -> Attempt:
        user = prompts.revise_message(problem.question, attempt.steps, attempt.answer, feedback)
        return self._attempt(self._ask(prompts.REVISE, user))

    def critique(self, problem: Problem, attempt: Attempt) -> str:
        out = self._ask(
            prompts.CRITIQUE, prompts.critique_message(problem.question, attempt.steps, attempt.answer)
        )
        self.usage.add(out)
        return out.text.strip()

    # -- one step at a time ---------------------------------------------------

    def next_step(self, problem: Problem, steps: tuple[str, ...]) -> tuple[str | None, str | None]:
        """Return (next step, None) or (None, final answer)."""
        out = self._ask(prompts.NEXT_STEP, prompts.partial_message(problem.question, steps))
        self.usage.add(out)
        line = out.text.strip().splitlines()[0] if out.text.strip() else ""
        m = _STEP_RE.match(line)
        if m:
            return m.group(2), None
        answer = _ANSWER_RE.search(line)
        return None, (answer.group(1) if answer else line)

    # -- Tree of Thoughts -----------------------------------------------------

    def propose(self, problem: Problem, items: list[Item], k: int) -> list[list[Item]]:
        """Ask for up to ``k`` next moves; return the resulting states (unparseable moves are dropped)."""
        left = " ".join(exact(v) for v, _ in items)
        out = self._ask(prompts.PROPOSE, prompts.numbers_message(problem.question, left, k))
        self.usage.add(out)
        children: list[list[Item]] = []
        for line in out.text.splitlines():
            move = parse_move_line(line)
            if move is None:
                continue
            a, op, b, claimed = move
            found = find_move(items, a, op, b)
            if found is None:
                continue  # the model used a number it does not have
            i, _, j = found
            rest = [it for n, it in enumerate(items) if n not in (i, j)]
            children.append([*rest, (claimed, f"({items[i][1]} {op} {items[j][1]})")])
        return children

    def value(self, problem: Problem, items: list[Item]) -> str:
        left = " ".join(exact(v) for v, _ in items)
        out = self._ask(prompts.VALUE, prompts.numbers_message(problem.question, left))
        self.usage.add(out)
        word = out.text.strip().lower()
        return word if word in ("sure", "likely", "impossible") else "likely"


def make_reasoner(config: SimConfig | None = None, *, seed: int = 0, provider: str | None = None) -> Reasoner:
    """Default: FakeProvider + simulator. ``provider='ollama'|'openai'`` (or $LLM_PROVIDER): a real model."""
    chosen = (provider or os.environ.get("LLM_PROVIDER") or "fake").lower()
    if chosen == "fake":
        return Reasoner(
            FakeProvider(responder=SimulatedReasoner(config, seed=seed), model="simulated-reasoner")
        )
    return Reasoner(get_provider(chosen))


def simulator_of(reasoner: Reasoner) -> SimulatedReasoner:
    """The simulator behind a fake reasoner (used by the RL toy and the tests)."""
    responder = getattr(reasoner.provider, "_responder", None)
    if not isinstance(responder, SimulatedReasoner):
        raise TypeError("this reasoner is not backed by the simulator")
    return responder
