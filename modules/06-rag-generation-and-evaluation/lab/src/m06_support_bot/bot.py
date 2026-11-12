"""The support chatbot: guardrails -> routing -> rewrite -> filtered retrieval -> generation -> checks.

Every switch in :class:`BotConfig` exists so that an ablation can turn one
defence off and the evaluation can show what that defence was worth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from .guardrails import check_output, is_safety_risk, redact_pii, wants_human
from .kb import Chunk, build_chunks, load_documents
from .prompting import NO_ANSWER, build_messages, citations
from .providers import FakeProvider, LLMProvider, Message, get_provider
from .responder import ExtractiveResponder
from .retriever import BM25Index, Filters
from .rewrite import rewrite_query
from .text import content_tokens, detect_language, jaccard, split_units

REPLIES: dict[str, dict[str, str]] = {
    "no_answer": {
        "en": "I couldn't find that in the Hearthwise help centre, and I don't want to guess. "
        "Would you like me to pass you to a support agent?",
        "es": "No he encontrado esa información en el centro de ayuda de Hearthwise y no quiero adivinar. "
        "¿Quieres que te pase con una persona del equipo de soporte?",
    },
    "blocked": {
        "en": "I can't answer that safely here. Let me connect you with a support agent.",
        "es": "No puedo responder a eso de forma segura por aquí. "
        "Te paso con una persona del equipo de soporte.",
    },
    "human_request": {
        "en": "Of course. I'm connecting you with a support agent now.",
        "es": "Claro. Te paso ahora con una persona del equipo de soporte.",
    },
    "safety": {
        "en": "For your safety, unplug the device now and keep away from it. "
        "I'm connecting you with a support agent straight away.",
        "es": "Por seguridad, desenchufa el dispositivo ahora y mantente a distancia. "
        "Te paso ahora mismo con una persona del equipo de soporte.",
    },
}


@dataclass(frozen=True)
class BotConfig:
    """Every defence is a switch, so ablations can measure what it is worth."""

    name: str = "full"
    top_k: int = 4
    threshold: float = 0.5
    status_filter: bool = True
    audience_filter: bool = True
    sanitize: bool = True
    output_guardrail: bool = True
    rewrite: bool = True
    faq_route: bool = True
    faq_min_jaccard: float = 0.6
    gullible_model: bool = True

    def but(self, **changes: object) -> BotConfig:
        """A copy with some switches changed: ``BotConfig().but(sanitize=False)``."""
        return replace(self, **changes)  # type: ignore[arg-type]


ABLATIONS: dict[str, BotConfig] = {
    "full": BotConfig(),
    "no_status_filter": BotConfig(name="no_status_filter", status_filter=False),
    "no_audience_filter": BotConfig(name="no_audience_filter", audience_filter=False),
    "no_sanitiser": BotConfig(name="no_sanitiser", sanitize=False),
    "no_sanitiser_no_guardrail": BotConfig(name="no_sanitiser_no_guardrail", sanitize=False,
                                           output_guardrail=False),
    "no_rewrite": BotConfig(name="no_rewrite", rewrite=False),
}


@dataclass
class BotAnswer:
    """Everything that happened for one question: the reply and the trace behind it."""

    question: str
    redacted_question: str
    language: str
    route: str  # faq | rag | human | safety
    reason: str  # answered | no_answer | blocked | human_request | safety
    answer: str
    citations: list[str]
    context: list[Chunk] = field(default_factory=list)
    search_terms: list[str] = field(default_factory=list)
    redactions: tuple[str, ...] = ()
    guardrail: list[str] = field(default_factory=list)
    raw_output: str = ""
    evidence_score: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    messages: list[Message] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        """True when the bot did not answer from the knowledge base (handoff instead)."""
        return self.reason != "answered"


class SupportBot:
    """Hearthwise's grounded support assistant."""

    def __init__(self, config: BotConfig | None = None, provider: LLMProvider | None = None) -> None:
        self.config = config or BotConfig()
        self.docs = load_documents(sanitize=self.config.sanitize)
        self.chunks = build_chunks(self.docs)
        self.index = BM25Index(self.chunks)
        self.filters = Filters(
            audiences=frozenset({"public"}) if self.config.audience_filter else None,
            statuses=frozenset({"current"}) if self.config.status_filter else None,
        )
        self.faq = [c for c in self.chunks if c.doc.doc_type == "faq" and c.section.endswith("?")
                    and self.filters.allows(c)]
        self.responder: ExtractiveResponder | None = None
        self.provider = provider or self._default_provider()

    def _default_provider(self) -> LLMProvider:
        name = (os.environ.get("LLM_PROVIDER") or "fake").strip().lower()
        if name != "fake":
            return get_provider(name)
        self.responder = ExtractiveResponder(
            self.index.idf, self.index.max_idf, threshold=self.config.threshold,
            gullible=self.config.gullible_model,
        )
        return FakeProvider(responder=self.responder, model="extractive-1")

    # ------------------------------------------------------------------ routing
    def match_faq(self, question: str) -> Chunk | None:
        """Return the FAQ entry whose question is almost the same as the customer's."""
        q = set(content_tokens(question))
        best, best_sim = None, 0.0
        for chunk in self.faq:
            sim = jaccard(q, set(content_tokens(chunk.section)))
            if sim > best_sim:
                best, best_sim = chunk, sim
        return best if best_sim >= self.config.faq_min_jaccard else None

    def _reply(self, base: BotAnswer, reason: str) -> BotAnswer:
        base.reason = reason
        base.answer = REPLIES[reason][base.language]
        base.citations = []
        return base

    # --------------------------------------------------------------------- ask
    def ask(self, question: str) -> BotAnswer:
        red = redact_pii(question)
        q = red.text
        lang = detect_language(q)
        out = BotAnswer(question, q, lang, "rag", "answered", "", [], redactions=red.found)
        if is_safety_risk(q):
            out.route = "safety"
            return self._reply(out, "safety")
        if wants_human(q):
            out.route = "human"
            return self._reply(out, "human_request")
        faq = self.match_faq(q) if self.config.faq_route else None
        if faq is not None:
            out.route, out.context = "faq", [faq]
            out.raw_output = " ".join(f"{u} [{faq.doc_id}]" for u in split_units(faq.text))
        else:
            plain = list(dict.fromkeys(content_tokens(q)))
            out.search_terms = rewrite_query(q) if self.config.rewrite else plain
            hits = self.index.search(out.search_terms, k=self.config.top_k, filters=self.filters)
            out.context = [h.chunk for h in hits]
            if not out.context:
                return self._reply(out, "no_answer")
            out.messages = build_messages(q, out.context, lang)
            completion = self.provider.complete(out.messages, temperature=0.0, max_tokens=300, seed=0)
            out.raw_output = completion.text.strip()
            out.prompt_tokens, out.completion_tokens = completion.prompt_tokens, completion.completion_tokens
            if self.responder is not None:
                out.evidence_score = round(self.responder.last_trace.best_coverage, 3)
        if not out.raw_output or out.raw_output.startswith(NO_ANSWER):
            return self._reply(out, "no_answer")
        if self.config.output_guardrail:
            out.guardrail = check_output(out.raw_output)
            if out.guardrail:
                return self._reply(out, "blocked")
        out.answer = out.raw_output
        out.citations = citations(out.answer)
        return out
