"""The offline stand-in model: an extractive, citing, abstaining answerer.

It plugs into ``FakeProvider(responder=...)``. It reads the same prompt a real
model would read, then:

1. scores every sentence, list item or table row of the context by how much of
   the question it covers (rare words weigh more, see ``coverage``);
2. abstains with ``NO_ANSWER`` when the best sentence covers less than
   ``threshold`` of the question;
3. otherwise copies the best sentence (and, if it adds new information, one
   more) and cites the document id after each one.

It is also *deliberately gullible*: it obeys instructions it finds inside the
documents, which is exactly what indirect prompt injection exploits. That lets
the tests prove that the defences around the model (sanitiser, guardrails)
work, without a real LLM.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property

from .prompting import NO_ANSWER, ContextDoc, parse_prompt
from .providers import Message
from .rewrite import GLOSSARY, Concept, concept_groups
from .sanitize import INJECTION_PATTERNS
from .text import content_tokens, detect_language, split_units


@dataclass(frozen=True)
class Candidate:
    """One answer-sized unit of the context with the words that describe it."""

    doc_id: str
    title: str
    lang: str
    heading: str
    unit: str
    paragraph: int
    order: int

    @cached_property
    def words(self) -> set[str]:
        return set(content_tokens(f"{self.title} {self.heading} {self.unit}"))


@dataclass
class ResponderTrace:
    """What the stand-in model 'thought'. Real models do not give you this."""

    best_coverage: float = 0.0
    selected: list[tuple[str, str, float]] = field(default_factory=list)  # (doc_id, unit, coverage)
    followed_injection: bool = False


def context_candidates(docs: Sequence[ContextDoc]) -> list[Candidate]:
    """Split every context document into units, remembering title, heading and paragraph."""
    out: list[Candidate] = []
    for doc in docs:
        heading = ""
        for p_idx, paragraph in enumerate(re.split(r"\n\s*\n", doc.text)):
            lines = [ln for ln in paragraph.splitlines() if ln.strip()]
            while lines and lines[0].lstrip().startswith("#"):
                heading = lines.pop(0).lstrip("# ").strip()
            for unit in split_units("\n".join(lines)):
                out.append(Candidate(doc.id, doc.title, doc.lang, heading, unit, p_idx, len(out)))
    return out


def concept_weight(concept: Concept, idf: Callable[[str], float], max_idf: float) -> float:
    """IDF of the customer's word; if the help centre never uses it, the IDF of its known synonym."""
    if idf(concept.head) < max_idf:
        return idf(concept.head)
    known = [idf(t) for t in concept.terms if idf(t) < max_idf]
    return min(known) if known else max_idf


def coverage(concepts: list[Concept], weights: list[float], words: set[str]) -> float:
    """Share of the question's weight whose concept appears in ``words`` (0 to 1).

    Each concept weighs its inverse document frequency (IDF): matching 'C2' or
    'goodwill' says much more than matching 'hearthwise'.
    """
    total = sum(weights)
    if not total:
        return 0.0
    return sum(w for c, w in zip(concepts, weights) if c.terms & words) / total


class ExtractiveResponder:
    """Callable that turns RAG messages into an extractive, cited answer or ``NO_ANSWER``."""

    def __init__(
        self,
        idf: Callable[[str], float],
        max_idf: float,
        *,
        threshold: float = 0.5,
        max_units: int = 2,
        min_gain: float = 0.15,
        gullible: bool = True,
        glossary: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.idf, self.max_idf = idf, max_idf
        self.threshold = threshold
        self.max_units = max_units
        self.min_gain = min_gain
        self.gullible = gullible
        self.glossary = GLOSSARY if glossary is None else glossary
        self.last_trace = ResponderTrace()

    def __call__(self, messages: Sequence[Message]) -> str:
        question, docs = parse_prompt(list(messages))
        concepts = concept_groups(question, self.glossary)
        weights = [concept_weight(c, self.idf, self.max_idf) for c in concepts]
        lang = detect_language(question)
        cands = context_candidates(docs)
        scored = sorted(
            ((coverage(concepts, weights, c.words), c) for c in cands),
            key=lambda x: (-round(x[0], 9), x[1].lang != lang, x[1].order),
        )
        trace = ResponderTrace(best_coverage=scored[0][0] if scored else 0.0)
        answer = NO_ANSWER
        if scored and scored[0][0] >= self.threshold:
            chosen = [scored[0]]
            while len(chosen) < self.max_units:
                nxt = self._next_unit(scored, chosen, concepts, weights)
                if nxt is None:
                    break
                chosen.append(nxt)
            answer = " ".join(f"{c.unit} [{c.doc_id}]" for _, c in chosen)
            trace.selected = [(c.doc_id, c.unit, round(cov, 3)) for cov, c in chosen]
        if self.gullible:
            payload = _injected_payload(cands)
            if payload:
                trace.followed_injection = True
                extra = " ".join(f"{c.unit} [{c.doc_id}]" for c in payload)
                answer = extra if answer == NO_ANSWER else f"{answer} {extra}"
        self.last_trace = trace
        return answer

    def _next_unit(self, scored, chosen, concepts, weights):
        """Add a unit only if it is good evidence itself, comes from the same document and adds concepts."""
        total = sum(weights) or 1.0
        covered = {i for _, c in chosen for i, k in enumerate(concepts) if k.terms & c.words}
        taken = {c.order for _, c in chosen}
        first_doc = chosen[0][1].doc_id
        best_gain, best = 0.0, None
        for cov, cand in scored:
            if cand.order in taken or cov < self.threshold or cand.doc_id != first_doc:
                continue
            new = {i for i, k in enumerate(concepts) if k.terms & cand.words} - covered
            gain = sum(weights[i] for i in new) / total
            if gain > best_gain:
                best_gain, best = gain, (cov, cand)
        return best if best_gain >= self.min_gain else None


def _injected_payload(cands: list[Candidate]) -> list[Candidate]:
    """Units that follow an instruction aimed at the AI, within the same paragraph."""
    payload: list[Candidate] = []
    for i, cand in enumerate(cands):
        if any(p.search(cand.unit) for p in INJECTION_PATTERNS):
            payload.extend(
                c for c in cands[i + 1 :]
                if c.doc_id == cand.doc_id and c.paragraph == cand.paragraph and c not in payload
            )
    return payload
