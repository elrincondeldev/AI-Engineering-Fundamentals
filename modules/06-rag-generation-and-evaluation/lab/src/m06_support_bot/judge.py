"""Calibrate a faithfulness judge before trusting it (lesson 6.3).

We build a small labelled set from the bot's own answers. Each answer is kept
as is and also paraphrased (label: faithful), then corrupted in three ways whose
label we know by construction (unfaithful):

- ``number_swap``    one number changes ('30 days' -> '31 days');
- ``foreign_claim``  a sentence from an unrelated document is appended;
- ``negation``       the meaning is flipped ('cannot' -> 'can', 'is' -> 'is not').

Then we measure how often a judge agrees with those labels. The same procedure
calibrates an LLM-as-a-judge: replace the heuristic with the model and compare.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass

from .bot import SupportBot
from .data import read_jsonl
from .kb import Chunk
from .metrics import faithfulness
from .prompting import CITATION_RE
from .providers import LLMProvider, Message
from .text import detect_language, split_units

KINDS = ("original", "paraphrase", "number_swap", "foreign_claim", "negation")
FAITHFUL_KINDS = ("original", "paraphrase")


@dataclass(frozen=True)
class JudgeCase:
    qid: str
    kind: str
    answer: str
    context: tuple[Chunk, ...]
    faithful: bool  # the label, known by construction


_NUM_RE = re.compile(r"(?<![\w.:-])(\d+)(\.\d+)?(?![\w:-]|\.\d)")


def swap_number(answer: str) -> str | None:
    """Add 1 to the first standalone number outside citations and dates."""
    for m in _NUM_RE.finditer(answer):
        if CITATION_RE.search(answer[max(0, m.start() - 4): m.end() + 1]):
            continue
        new = str(int(m.group(1)) + 1) + (m.group(2) or "")
        return answer[: m.start()] + new + answer[m.end():]
    return None


_NEGATIONS = (
    (re.compile(r"\bcannot\b"), "can"),
    (re.compile(r"\bnever\b"), "always"),
    (re.compile(r"\bno\b"), "a"),
    (re.compile(r"\b(is|are|can|will|does)\b"), r"\1 not"),
)


def negate(answer: str) -> str | None:
    """Flip the meaning of an English answer with the first rule that applies."""
    if detect_language(answer) != "en":
        return None
    for rx, repl in _NEGATIONS:
        if rx.search(answer):
            return rx.sub(repl, answer, count=1)
    return None


def paraphrase(answer: str) -> str:
    """Same facts, a few extra words, as a chatty model would write them."""
    spanish = detect_language(answer) == "es"
    prefix = "Según nuestro centro de ayuda, " if spanish else "According to our help centre, "
    return prefix + answer[0].lower() + answer[1:]


def build_cases(bot: SupportBot | None = None, *, seed: int = 6) -> list[JudgeCase]:
    """Labelled judge cases from the bot's answered golden questions."""
    bot = bot or SupportBot()
    rng = random.Random(seed)
    public = [c for c in bot.chunks if bot.filters.allows(c)]
    cases: list[JudgeCase] = []
    for row in read_jsonl("rag_eval/qa.jsonl"):
        out = bot.ask(row["question"])
        if out.reason != "answered":
            continue
        ctx = tuple(out.context)
        cases.append(JudgeCase(row["id"], "original", out.answer, ctx, True))
        cases.append(JudgeCase(row["id"], "paraphrase", paraphrase(out.answer), ctx, True))
        swapped = swap_number(out.answer)
        if swapped:
            cases.append(JudgeCase(row["id"], "number_swap", swapped, ctx, False))
        in_ctx = {c.doc_id for c in ctx}
        foreign = rng.choice([c for c in public if c.doc_id not in in_ctx])
        unit = split_units(foreign.text)[0]
        wrong = f"{out.answer} {unit} [{ctx[0].doc_id}]"  # a claim from elsewhere, cited as if supported
        cases.append(JudgeCase(row["id"], "foreign_claim", wrong, ctx, False))
        negated = negate(out.answer)
        if negated:
            cases.append(JudgeCase(row["id"], "negation", negated, ctx, False))
    return cases


Judge = Callable[[JudgeCase], bool]  # returns True when it thinks the answer is faithful


def heuristic_judge(tau: float = 0.5, check_numbers: bool = True) -> Judge:
    """Faithful only if every claim is supported (see metrics.check_claim)."""

    def judge(case: JudgeCase) -> bool:
        return faithfulness(case.answer, list(case.context), tau=tau, check_numbers=check_numbers) == 1.0

    return judge


@dataclass(frozen=True)
class Calibration:
    name: str
    cases: int
    agreement: float
    false_alarms: int  # faithful answers (original or paraphrase) the judge rejected
    faithful_cases: int
    caught: dict[str, tuple[int, int]]  # kind -> (caught, total) for the corrupted kinds


def calibrate(name: str, judge: Judge, cases: list[JudgeCase]) -> Calibration:
    """Compare the judge's verdicts with the known labels."""
    verdicts = [judge(c) for c in cases]
    agree = sum(v == c.faithful for v, c in zip(verdicts, cases))
    caught: dict[str, tuple[int, int]] = {}
    for kind in (k for k in KINDS if k not in FAITHFUL_KINDS):
        ks = [(v, c) for v, c in zip(verdicts, cases) if c.kind == kind]
        caught[kind] = (sum(not v for v, _ in ks), len(ks))
    false_alarms = sum(not v for v, c in zip(verdicts, cases) if c.faithful)
    n_faithful = sum(c.faithful for c in cases)
    return Calibration(name, len(cases), round(agree / len(cases), 3), false_alarms, n_faithful, caught)


JUDGE_PROMPT = """You check whether an answer is fully supported by the documents.
Reply with exactly one word: SUPPORTED if every statement in the answer is stated in the
documents, UNSUPPORTED otherwise. Numbers must match exactly.

<documents>
{documents}
</documents>

<answer>{answer}</answer>"""


def llm_judge(provider: LLMProvider) -> Judge:
    """An LLM-as-a-judge with the same interface. Calibrate it with ``calibrate`` before use."""

    def judge(case: JudgeCase) -> bool:
        docs = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in case.context)
        prompt = JUDGE_PROMPT.format(documents=docs, answer=case.answer)
        out = provider.complete([Message("user", prompt)], temperature=0.0, max_tokens=5, seed=0)
        return out.text.strip().upper().startswith("SUPPORTED")

    return judge
