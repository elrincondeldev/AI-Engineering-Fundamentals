"""Supervised fine-tuning (SFT) data: a chat template, a loss mask and a data cleaner.

The cleaner is scored against ``datasets/sft/labels.jsonl``, which flags the problem rows that
were planted on purpose.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .data import read_jsonl
from .text import ngrams, tokenize

# --------------------------------------------------------------------------- #
# Chat template and loss mask                                                 #
# --------------------------------------------------------------------------- #

ROLE_MARKERS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
END = "<|end|>"
_SPECIAL_RE = re.compile(r"(<\|[a-z]+\|>)")
_PIECE_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def render_chat(messages: Sequence[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Render chat messages into the single string the model actually sees.

    Each turn becomes ``<|role|>\\ncontent<|end|>\\n``. With ``add_generation_prompt`` the string
    ends with an open ``<|assistant|>`` turn, which is how you ask the model to answer.
    """
    parts = []
    for msg in messages:
        if msg["role"] not in ROLE_MARKERS:
            raise ValueError(f"unknown role {msg['role']!r}")
        parts.append(f"{ROLE_MARKERS[msg['role']]}\n{msg['content']}{END}\n")
    if add_generation_prompt:
        parts.append(f"{ROLE_MARKERS['assistant']}\n")
    return "".join(parts)


def row_to_messages(row: dict[str, str]) -> list[dict[str, str]]:
    """Turn one ``sft/instructions.jsonl`` row into chat messages."""
    msgs = []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
    msgs.append({"role": "user", "content": row["instruction"]})
    msgs.append({"role": "assistant", "content": row["response"]})
    return msgs


def chat_tokens(rendered: str) -> list[str]:
    """Split a rendered chat into tokens; each role marker and ``<|end|>`` is one special token."""
    out: list[str] = []
    for piece in _SPECIAL_RE.split(rendered):
        if _SPECIAL_RE.fullmatch(piece):
            out.append(piece)
        else:
            out.extend(_PIECE_RE.findall(piece))
    return out


def loss_mask(tokens: Sequence[str]) -> list[int]:
    """1 for tokens the model is trained to predict (assistant content and its ``<|end|>``), else 0.

    The system prompt, the user message and the role markers are context only: the model reads
    them but is never graded on predicting them.
    """
    mask: list[int] = []
    in_assistant = False
    for tok in tokens:
        if tok in ROLE_MARKERS.values():
            in_assistant = tok == ROLE_MARKERS["assistant"]
            mask.append(0)
        elif tok == END:
            mask.append(1 if in_assistant else 0)
            in_assistant = False
        else:
            mask.append(1 if in_assistant else 0)
    return mask


@dataclass(frozen=True)
class MaskStats:
    """How many tokens a set of examples contains and how many of them carry loss."""

    total_tokens: int
    trained_tokens: int

    @property
    def trained_share(self) -> float:
        return self.trained_tokens / self.total_tokens if self.total_tokens else 0.0


def mask_stats(rows: Sequence[dict[str, str]]) -> MaskStats:
    """Token counts for a list of SFT rows rendered with :func:`render_chat`."""
    total = trained = 0
    for row in rows:
        toks = chat_tokens(render_chat(row_to_messages(row)))
        total += len(toks)
        trained += sum(loss_mask(toks))
    return MaskStats(total, trained)


# --------------------------------------------------------------------------- #
# Data-quality checks                                                         #
# --------------------------------------------------------------------------- #

STOPWORDS = {
    "en": {"the", "and", "is", "you", "your", "to", "of", "it", "in", "for", "with", "are", "years", "days"},
    "es": {"el", "la", "los", "las", "es", "de", "que", "y", "tienes", "con", "para", "años", "días",
           "solo"},
    "de": {"der", "die", "das", "und", "ist", "beträgt", "jahre", "nicht", "mit", "für", "zwei"},
    "fr": {"le", "la", "les", "est", "et", "vous", "avec", "pour", "ans", "jours", "deux"},
}

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "card": re.compile(r"\bcard (?:ending|number)\s*(?:in\s*)?\d{4}\b", re.IGNORECASE),
    "street address": re.compile(
        r"\b(?:calle|avenida|street|road)\s+[a-záéíóúñ]+\s+\d+", re.IGNORECASE
    ),
}

# Claims that were true only under the superseded 2023 returns policy
# (datasets/kb/policies/returns-policy-2023.md). The 2025 policy says 30 days and no fee.
OUTDATED_CLAIMS = (
    re.compile(r"\b14 days\b", re.IGNORECASE),
    re.compile(r"\b10% restocking", re.IGNORECASE),
    re.compile(r"\b10 business days\b", re.IGNORECASE),
)


def detect_language(text: str) -> str:
    """Guess the language from stopwords; ``"unknown"`` when there is no evidence."""
    words = tokenize(text)
    scores = {lang: sum(w in sw for w in words) for lang, sw in STOPWORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "unknown"


def is_empty(row: dict[str, str]) -> bool:
    return not row["response"].strip()


def is_low_quality(row: dict[str, str], max_repeat_share: float = 0.5) -> bool:
    """Very repetitive responses such as ``"asdf asdf asdf"``: few distinct words."""
    words = tokenize(row["response"])
    if len(words) < 3:
        return False
    return len(set(words)) / len(words) <= max_repeat_share


def is_wrong_language(row: dict[str, str]) -> bool:
    detected = detect_language(row["response"])
    return detected not in ("unknown", row["lang"])


def has_pii(row: dict[str, str]) -> bool:
    return any(p.search(row["response"]) for p in PII_PATTERNS.values())


def is_outdated(row: dict[str, str]) -> bool:
    return any(p.search(row["response"]) for p in OUTDATED_CLAIMS)


@dataclass
class ContaminationIndex:
    """Word n-grams of every evaluation item, to spot training rows copied from the eval set.

    With ``with_answers=True`` an item is the question plus its reference answer, so only rows
    that copy both are flagged. With ``False`` an item is the question alone (a softer leak).
    """

    n: int = 8
    items: dict[str, set[tuple[str, ...]]] = field(default_factory=dict)

    @classmethod
    def from_rag_eval(cls, n: int = 8, with_answers: bool = True) -> ContaminationIndex:
        idx = cls(n=n)
        for q in read_jsonl("rag_eval/qa.jsonl"):
            text = f"{q['question']} {q['reference_answer']}" if with_answers else q["question"]
            toks = tokenize(text)
            idx.items[q["id"]] = set(ngrams(toks, min(n, len(toks))))
        return idx

    def matches(self, text: str, min_share: float = 0.5) -> list[str]:
        """Eval items with at least ``min_share`` of their n-grams inside ``text``."""
        toks = tokenize(text)
        found = []
        for item_id, item_grams in self.items.items():
            size = len(next(iter(item_grams))) if item_grams else 0
            grams = set(ngrams(toks, size)) if size else set()
            if item_grams and len(item_grams & grams) / len(item_grams) >= min_share:
                found.append(item_id)
        return found


def make_checks(index: ContaminationIndex) -> dict[str, Callable[[dict[str, str]], bool]]:
    """Checks in the order they run. The first one that fires decides the reason."""
    return {
        "empty_response": is_empty,
        "low_quality": is_low_quality,
        "wrong_language": is_wrong_language,
        "pii_in_response": has_pii,
        "contaminated_rag_eval": lambda row: bool(index.matches(f"{row['instruction']} {row['response']}")),
        "outdated_policy": is_outdated,
    }


def dedup_key(row: dict[str, str]) -> tuple[str, str]:
    """Exact-duplicate key: normalised instruction and response."""
    return " ".join(tokenize(row["instruction"])), " ".join(tokenize(row["response"]))


@dataclass
class CleanReport:
    """What the cleaner kept and why it dropped the rest."""

    kept: list[dict[str, str]]
    dropped: dict[str, str]  # row id -> reason

    def count(self, reason: str) -> int:
        return sum(r == reason for r in self.dropped.values())


def clean(rows: Sequence[dict[str, str]], index: ContaminationIndex | None = None) -> CleanReport:
    """Run every check, then drop exact duplicates among the rows that survived."""
    checks = make_checks(index or ContaminationIndex.from_rag_eval())
    kept: list[dict[str, str]] = []
    dropped: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        reason = next((name for name, check in checks.items() if check(row)), None)
        if reason is None:
            key = dedup_key(row)
            if key in seen:
                reason = "duplicate"
            seen.add(key)
        if reason:
            dropped[row["id"]] = reason
        else:
            kept.append(row)
    return CleanReport(kept, dropped)


@dataclass(frozen=True)
class FlagScore:
    """Precision and recall of one check against the planted labels."""

    flag: str
    labelled: int
    flagged: int
    correct: int

    @property
    def precision(self) -> float:
        return self.correct / self.flagged if self.flagged else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.labelled if self.labelled else 0.0


def score_against_labels(report: CleanReport, labels: dict[str, list[str]]) -> list[FlagScore]:
    """Compare the cleaner's reasons with ``sft/labels.jsonl`` (one score per flag).

    ``duplicate`` is scored separately by :func:`duplicates_resolved`: when two rows are identical
    it does not matter which copy the cleaner keeps.
    """
    flags = sorted({f for fl in labels.values() for f in fl} - {"duplicate"})
    scores = []
    for flag in flags:
        truth = {rid for rid, fl in labels.items() if flag in fl}
        found = {rid for rid, reason in report.dropped.items() if reason == flag}
        scores.append(FlagScore(flag, len(truth), len(found), len(truth & found)))
    return scores


def duplicates_resolved(
    report: CleanReport, rows: Sequence[dict[str, str]], labels: dict[str, list[str]]
) -> tuple[int, int]:
    """(labelled duplicates whose text survives exactly once, labelled duplicates)."""
    kept = Counter(dedup_key(r) for r in report.kept)
    dups = [r for r in rows if "duplicate" in labels.get(r["id"], [])]
    return sum(kept[dedup_key(r)] == 1 for r in dups), len(dups)


def question_leaks(rows: Sequence[dict[str, str]], n: int = 8, min_share: float = 0.5) -> list[str]:
    """Ids of rows whose instruction alone matches an evaluation question (answer not copied)."""
    index = ContaminationIndex.from_rag_eval(n=n, with_answers=False)
    return [r["id"] for r in rows if index.matches(r["instruction"], min_share)]


def load_sft() -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    """Rows of ``sft/instructions.jsonl`` and the planted labels, keyed by id."""
    rows = read_jsonl("sft/instructions.jsonl")
    labels = {row["id"]: row["flags"] for row in read_jsonl("sft/labels.jsonl")}
    return rows, labels


def response_repeats(rows: Sequence[dict[str, str]]) -> Counter[str]:
    """How many times each distinct response appears (a quick diversity check)."""
    return Counter(" ".join(tokenize(r["response"])) for r in rows)
