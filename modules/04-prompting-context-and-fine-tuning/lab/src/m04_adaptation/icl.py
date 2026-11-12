"""A toy in-context learner: a transparent stand-in for an LLM that *reads the prompt*.

It is plugged into ``FakeProvider(responder=...)``. Its rules are simple on purpose so that
you can predict and explain every answer:

1. Label descriptions: it scores each intent by the words the ticket shares with the
   intent's description in the system message (zero-shot behaviour).
2. Examples: every labelled example votes for its label with a weight equal to its word
   overlap (Jaccard) with the ticket (few-shot behaviour, like a nearest-neighbour vote).
3. It answers in JSON when the system message asks for JSON, adds a short "Reasoning:"
   line when asked to think step by step, and slips on the format for about 1 ticket in 20
   (chosen by a hash of the ticket) so that you can practise validation and retries.

A real LLM brings far more prior knowledge (it understands synonyms, other languages and
intent without examples). Use this toy to build and debug the *evaluation harness*; then
run the same harness against a real model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .providers import Message
from .textfeat import jaccard, stable_hash, stems

EXAMPLE_WEIGHT = 4.0  # how much one perfectly matching example counts versus one shared description word
CONTEXT_WEIGHT = 0.5  # words from the "Customer context:" line count half
SLIP_EVERY = 20  # about 1 first answer in 20 breaks the output format
RETRY_PREFIX = "Your previous answer"

_LABEL_LINE = re.compile(r"^\s*-\s*([a-z][a-z_]*[a-z])\s*(?::\s*(.*))?$")


@dataclass(frozen=True)
class ParsedPrompt:
    """What the toy learner understood from the messages."""

    labels: list[str]
    descriptions: dict[str, str]
    examples: list[tuple[str, str]]  # (example text, label)
    query: str
    context: str
    json_mode: bool
    reasoning: bool
    is_retry: bool


def split_user_content(content: str) -> tuple[str, str]:
    """Return (ticket text, customer context) from a user message built by ``prompts.py``."""
    ticket: list[str] = []
    context: list[str] = []
    for line in content.splitlines():
        head, _, rest = line.partition(":")
        if head.strip().lower() == "customer context":
            context.append(rest)
        elif head.strip().lower() == "ticket":
            ticket.append(rest)
        else:
            ticket.append(line)
    return " ".join(" ".join(ticket).split()), " ".join(" ".join(context).split())


def parse_label(text: str) -> str:
    """Read a label from an assistant message: JSON ``{"intent": ...}`` or a bare word."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return str(json.loads(text).get("intent", ""))
        except (json.JSONDecodeError, AttributeError):
            return ""
    return text


def parse_prompt(messages: Sequence[Message]) -> ParsedPrompt:
    """Pull labels, descriptions, examples and the query out of a chat prompt."""
    system = "\n".join(m.content for m in messages if m.role == "system")
    labels: list[str] = []
    descriptions: dict[str, str] = {}
    in_intents = False
    for line in system.splitlines():
        if line.strip().lower().startswith("intents"):
            in_intents = True
            continue
        m = _LABEL_LINE.match(line) if in_intents else None
        if m is None:
            in_intents = False
            continue
        labels.append(m.group(1))
        descriptions[m.group(1)] = m.group(2) or ""
    convo = [m for m in messages if m.role != "system"]
    is_retry = bool(convo) and convo[-1].role == "user" and convo[-1].content.startswith(RETRY_PREFIX)
    examples: list[tuple[str, str]] = []
    query_idx = max((i for i, m in enumerate(convo) if m.role == "user" and "Ticket:" in m.content), default=-1)
    for i in range(query_idx):
        if convo[i].role == "user" and i + 1 < query_idx and convo[i + 1].role == "assistant":
            text, ctx = split_user_content(convo[i].content)
            examples.append((f"{text} {ctx}".strip(), parse_label(convo[i + 1].content)))
    query, context = split_user_content(convo[query_idx].content) if query_idx >= 0 else ("", "")
    return ParsedPrompt(
        labels=labels,
        descriptions=descriptions,
        examples=examples,
        query=query,
        context=context,
        json_mode="json" in system.lower(),
        reasoning="step by step" in system.lower(),
        is_retry=is_retry,
    )


@dataclass
class ToyInContextLearner:
    """Callable responder for ``FakeProvider``. See the module docstring for its rules.

    ``position_bias`` (0 to 1) lowers the weight of examples in the middle of the prompt,
    imitating the "lost in the middle" effect so that you can test a position probe.
    """

    example_weight: float = EXAMPLE_WEIGHT
    position_bias: float = 0.0
    slips: bool = True
    last_scores: dict[str, float] = field(default_factory=dict, repr=False)

    def position_weight(self, index: int, n: int) -> float:
        if n < 2 or self.position_bias == 0:
            return 1.0
        centre_distance = abs(2 * index / (n - 1) - 1)  # 1 at both edges, 0 in the middle
        return 1.0 - self.position_bias * (1.0 - centre_distance)

    def scores(self, parsed: ParsedPrompt) -> dict[str, float]:
        """Score every label: shared description words plus similarity-weighted example votes."""
        q = stems(parsed.query)
        ctx = stems(parsed.context)
        full = q | ctx
        out: dict[str, float] = {}
        for label in parsed.labels:
            desc = stems(parsed.descriptions.get(label, "") + " " + label.replace("_", " "))
            out[label] = len(q & desc) + CONTEXT_WEIGHT * len(ctx & desc)
        n = len(parsed.examples)
        for i, (text, label) in enumerate(parsed.examples):
            if label in out:
                out[label] += self.example_weight * self.position_weight(i, n) * jaccard(full, stems(text))
        return out

    def predict(self, parsed: ParsedPrompt) -> str:
        """Highest score wins; ties go to the label listed first (a real 'default answer' bias)."""
        self.last_scores = self.scores(parsed)
        if not self.last_scores:
            return ""
        return max(parsed.labels, key=lambda label: self.last_scores[label])

    def __call__(self, messages: Sequence[Message]) -> str:
        parsed = parse_prompt(messages)
        label = self.predict(parsed)
        answer = json.dumps({"intent": label}) if parsed.json_mode else label
        h = stable_hash(parsed.query)
        if self.slips and not parsed.is_retry and h % SLIP_EVERY == 0:
            pretty = label.replace("_", " ").title()
            if (h // SLIP_EVERY) % 2 == 0:
                answer = f"Sure! Here is the classification: {answer}"  # recoverable by a lenient parser
            else:
                answer = json.dumps({"intent": pretty}) if parsed.json_mode else pretty  # needs a retry
        if parsed.reasoning:
            shared = sorted(stems(parsed.query) & stems(" ".join(parsed.descriptions.values())))[:4]
            votes = sum(1 for _, ex_label in parsed.examples if ex_label == label)
            answer = (
                f"Reasoning: the ticket shares the words {', '.join(shared) or 'none'} with the intent list; "
                f"{votes} of {len(parsed.examples)} examples are labelled {label}.\nAnswer: {answer}"
            )
        return answer
