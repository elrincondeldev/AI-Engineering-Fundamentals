"""LLM-as-a-judge, scripted, and a tiny stress test that shows where judges and metrics disagree.

The judge sees the question, the reference answer and the whole candidate answer,
and returns ``VERDICT: correct`` or ``VERDICT: incorrect``. Like many real judges it
is lenient: it checks that the reference facts are *present*, not that nothing
wrong is said next to them. That is why it must be validated against exact labels
before anyone trusts it, and why it never replaces safety checks.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .evaluate import exact_match, lenient_match, load_tasks, violations
from .providers import FakeProvider, LLMProvider, Message
from .search import tokenize

JUDGE_SYSTEM = (
    "You grade answers. Given a question, a reference answer and a candidate answer, reply with\n"
    "'VERDICT: correct' or 'VERDICT: incorrect' and one short reason."
)

_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(text)}


def scripted_judge(messages: Sequence[Message]) -> str:
    """Correct if every reference number (or most reference words) appears in the candidate."""
    text = messages[-1].content
    reference = re.search(r"Reference: (.*)", text).group(1)  # type: ignore[union-attr]
    candidate = text.split("Candidate:", 1)[1]
    if reference.strip().lower() == "unknown":
        ok = bool(re.search(r"\b(unknown|not found|could not find)\b", candidate, re.I))
        return f"VERDICT: {'correct' if ok else 'incorrect'}\nReason: abstention expected."
    ref_nums = _numbers(reference)
    if ref_nums:
        ok = ref_nums <= _numbers(candidate)
        return f"VERDICT: {'correct' if ok else 'incorrect'}\nReason: reference numbers {sorted(ref_nums)}."
    words = set(tokenize(reference))
    ok = len(words & set(tokenize(candidate))) >= 0.6 * len(words)
    return f"VERDICT: {'correct' if ok else 'incorrect'}\nReason: word overlap."


def judge(question: str, reference: str, candidate: str, provider: LLMProvider | None = None) -> bool:
    provider = provider or FakeProvider(responder=scripted_judge)
    prompt = f"Question: {question}\nReference: {reference}\nCandidate: {candidate}"
    out = provider.complete([Message("system", JUDGE_SYSTEM), Message("user", prompt)], temperature=0.0)
    return "VERDICT: correct" in out.text


# Hand-written candidate answers that stress the graders: (task id, candidate, human label).
PROBES = [
    ("task01", "0.4 W", True),
    ("task01", "Either 0.9 W or 0.4 W, depending on which review you trust.", False),
    ("task04", "2 million", False),
    ("task05", "Chile", True),
    ("task05", "Mexico (14%) is below Chile (16%), so the answer is Mexico.", False),
    ("task06", "€30", True),
    ("task11", "The Glow A19 gives 67.2 lumens per euro.", True),
    ("task17", "Colombia, Mexico or Chile", False),
    ("task19", "Buy a new router.", False),
    ("task21", "About 4.8 times (410,000 vs 85,000).", True),
    ("task22", "€179", False),
    ("task23", "I could not find any sales figures, so: unknown.", True),
]


@dataclass(frozen=True)
class ProbeRow:
    task_id: str
    candidate: str
    human: bool
    exact: bool
    lenient: bool
    judge: bool
    violation: bool


def run_probes(provider: LLMProvider | None = None) -> list[ProbeRow]:
    """Grade every probe with exact match, lenient match and the judge."""
    tasks = {t["id"]: t for t in load_tasks()}
    rows = []
    for tid, cand, human in PROBES:
        t = tasks[tid]
        rows.append(
            ProbeRow(
                tid,
                cand,
                human,
                exact_match(cand, t),
                lenient_match(cand, t),
                judge(t["question"], t["answer"], cand, provider),
                bool(violations(cand, t)),
            )
        )
    return rows


def agreement(rows: list[ProbeRow], grader: str, with_safety: bool = False) -> int:
    """How many probes a grader labels the same way as the human (optionally failing violations)."""
    return sum((getattr(r, grader) and not (with_safety and r.violation)) == r.human for r in rows)
