"""Find personal data (PII) and replace it with placeholders before a prompt leaves the building.

Three redactor levels, so you can measure what each idea buys you:

- ``naive``  emails + "any long run of digits" as a phone number;
- ``regex``  emails + international phone numbers (+CC ...);
- ``names``  ``regex`` + a capitalised-word-pair heuristic for person names.

All levels also *pseudonymise* account identifiers (order and serial numbers) so the model
can still say "your order <ORDER_1>" and the playground can put the real value back locally.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .data import iter_csv

DIRECT = ("EMAIL", "PHONE", "NAME")  # direct identifiers: evaluated with precision and recall
LEVELS = ("naive", "regex", "names")

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"\+\d{1,3}(?:[ .-]?\d{2,4}){2,5}")
NAIVE_PHONE_RE = re.compile(r"\d[\d ()-]{7,}\d")
ORDER_RE = re.compile(r"#HW-\d{6}", re.IGNORECASE)
SERIAL_RE = re.compile(r"\bSN\d{8}\b", re.IGNORECASE)

_UP, _LOW = "A-ZÁÉÍÓÚÑÜ", "a-záéíóúñü"
NAME_RE = re.compile(rf"\b[{_UP}][{_LOW}]+ [{_UP}][{_LOW}]+\b")
# Capitalised words that are not person names in this domain: products, brand, greetings.
NOT_NAMES = {
    "Hearthwise",
    "Hearth",
    "Hub",
    "Ember",
    "Glow",
    "Spark",
    "Mini",
    "Watcher",
    "Lookout",
    "Plus",
    "Support",
    "HomeMesh",
    "Hola",
    "Hello",
    "Hi",
    "Hey",
    "Buenas",
    "Good",
    "Best",
    "Thanks",
    "Thank",
    "Dear",
    "Regards",
    "Gracias",
    "Cheers",
    "Sent",
    "From",
}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    text: str


def _spans(pattern: re.Pattern[str], kind: str, text: str) -> list[Span]:
    return [Span(m.start(), m.end(), kind, m.group(0)) for m in pattern.finditer(text)]


def find_names(text: str) -> list[Span]:
    """Two capitalised words in a row that are not product or greeting words."""
    return [s for s in _spans(NAME_RE, "NAME", text) if not any(w in NOT_NAMES for w in s.text.split())]


def detect(text: str, level: str = "names") -> list[Span]:
    """All PII spans for a redactor level, sorted and without overlaps (earlier, longer wins)."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; use one of {LEVELS}")
    found = _spans(EMAIL_RE, "EMAIL", text)
    found += _spans(NAIVE_PHONE_RE if level == "naive" else PHONE_RE, "PHONE", text)
    found += _spans(ORDER_RE, "ORDER", text) + _spans(SERIAL_RE, "SERIAL", text)
    if level == "names":
        found += find_names(text)
    found.sort(key=lambda s: (s.start, -(s.end - s.start)))
    merged: list[Span] = []
    for span in found:
        if merged and span.start < merged[-1].end:
            continue
        merged.append(span)
    return merged


@dataclass
class Pseudonymiser:
    """Replaces PII with stable placeholders and remembers the mapping (keep it local!)."""

    level: str = "names"
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original
    _reverse: dict[str, str] = field(default_factory=dict)  # original -> placeholder
    _counts: Counter[str] = field(default_factory=Counter)

    def redact(self, text: str) -> tuple[str, Counter[str]]:
        """Return the redacted text and how many items of each kind were replaced."""
        out, last = [], 0
        seen: Counter[str] = Counter()
        for span in detect(text, self.level):
            placeholder = self._reverse.get(span.text)
            if placeholder is None:
                self._counts[span.kind] += 1
                placeholder = f"<{span.kind}_{self._counts[span.kind]}>"
                self._reverse[span.text] = placeholder
                self.mapping[placeholder] = span.text
            out.append(text[last : span.start] + placeholder)
            last = span.end
            seen[span.kind] += 1
        out.append(text[last:])
        return "".join(out), seen

    def restore(self, text: str) -> str:
        """Put the original values back into a model answer, on our side of the boundary."""
        for placeholder, original in self.mapping.items():
            text = text.replace(placeholder, original)
        return text


# --------------------------------------------------------------------------- #
# Evaluation on datasets/tickets                                              #
# --------------------------------------------------------------------------- #

SIGNOFF_RE = re.compile(r"^best regards$", re.IGNORECASE | re.MULTILINE)


def ticket_payload(row: dict[str, str]) -> str:
    """What a support agent pastes into the playground: the sender line plus the ticket body."""
    return f"From: {row['customer_email']}\n\n{row['body']}"


def gold_spans(payload: str) -> list[Span]:
    """Reference labels for direct identifiers, from the dataset's known layout.

    The sender email is on the ``From:`` line. Every signature in this dataset is
    "Best regards" + a name line + a phone line. Independent of the redactor's regexes,
    so the evaluation is not graded by the thing it grades.
    """
    first = payload.split("\n", 1)[0]
    gold = [Span(6, len(first), "EMAIL", first[6:])]
    for m in SIGNOFF_RE.finditer(payload):
        rest = payload[m.end() + 1 :].split("\n")
        offset = m.end() + 1
        for kind, line in zip(("NAME", "PHONE"), rest[:2]):
            gold.append(Span(offset, offset + len(line), kind, line))
            offset += len(line) + 1
    return gold


def _covered(gold: Span, spans: Sequence[Span]) -> bool:
    """A gold item counts as caught when every letter and digit of it is inside a redacted span."""
    for i in range(gold.start, gold.end):
        if gold.text[i - gold.start].isalnum() and not any(s.start <= i < s.end for s in spans):
            return False
    return True


@dataclass
class LevelReport:
    level: str
    gold: Counter[str] = field(default_factory=Counter)
    caught: Counter[str] = field(default_factory=Counter)
    false_positives: Counter[str] = field(default_factory=Counter)
    leaking_tickets: int = 0  # tickets with at least one direct identifier left in clear text
    pseudonymised: Counter[str] = field(default_factory=Counter)  # ORDER / SERIAL replacements
    missed_examples: list[tuple[str, str, str]] = field(default_factory=list)  # (ticket, kind, text)
    fp_examples: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return sum(self.caught.values()) / max(sum(self.gold.values()), 1)

    @property
    def precision(self) -> float:
        tp = sum(self.caught.values())
        return tp / max(tp + sum(self.false_positives.values()), 1)


def evaluate_level(rows: Iterable[dict[str, str]], level: str) -> LevelReport:
    report = LevelReport(level)
    for row in rows:
        payload = ticket_payload(row)
        gold = gold_spans(payload)
        spans = detect(payload, level)
        direct = [s for s in spans if s.kind in DIRECT]
        leaked = False
        for g in gold:
            report.gold[g.kind] += 1
            if _covered(g, spans):
                report.caught[g.kind] += 1
            else:
                leaked = True
                report.missed_examples.append((row["ticket_id"], g.kind, g.text))
        report.leaking_tickets += leaked
        for s in direct:
            if not any(s.start < g.end and g.start < s.end for g in gold):
                report.false_positives[s.kind] += 1
                report.fp_examples.append((row["ticket_id"], s.kind, s.text))
        for s in spans:
            if s.kind not in DIRECT:
                report.pseudonymised[s.kind] += 1
    return report


def load_tickets() -> list[dict[str, str]]:
    return list(iter_csv("tickets/tickets.csv"))


def evaluate_all(levels: Sequence[str] = LEVELS) -> list[LevelReport]:
    rows = load_tickets()
    return [evaluate_level(rows, level) for level in levels]
