"""Context engineering: assemble a prompt from prioritised pieces under a token budget.

Pipeline: compact (clean, deduplicate) -> order (stable prefix first, query last) -> prune
(drop the lowest-priority optional pieces until the prompt fits). Every dropped piece is
reported with a reason, so that nothing disappears silently.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from .prompts import ExampleSelector, PromptConfig, answer_message, system_prompt, user_message
from .providers import Message, count_tokens
from .textfeat import jaccard, stems
from .tickets import Ticket

Kind = Literal["system", "example", "profile", "query"]
MESSAGE_OVERHEAD = 4  # tokens a chat format spends on role markers per message (approximation)


@dataclass(frozen=True)
class ContextItem:
    """One candidate piece of context. Higher ``priority`` is kept first; ``required`` is never dropped."""

    kind: Kind
    text: str
    priority: float
    required: bool = False
    stable: bool = False  # identical for every request: belongs to the cacheable prefix
    answer: str = ""  # examples only: the assistant reply that follows the example ticket

    @property
    def tokens(self) -> int:
        n = count_tokens(self.text) + MESSAGE_OVERHEAD
        return n + count_tokens(self.answer) + MESSAGE_OVERHEAD if self.kind == "example" else n


@dataclass
class Assembly:
    """The assembled messages and an audit trail of what was kept and dropped."""

    messages: list[Message]
    kept: list[ContextItem]
    dropped: list[tuple[ContextItem, str]] = field(default_factory=list)
    tokens_offered: int = 0

    @property
    def tokens(self) -> int:
        return sum(item.tokens for item in self.kept)

    @property
    def examples_kept(self) -> int:
        return sum(item.kind == "example" for item in self.kept)


def drop_near_duplicates(items: list[ContextItem], threshold: float = 0.9) -> tuple[list[ContextItem], list[ContextItem]]:
    """Drop examples that say almost the same thing as a higher-priority example already kept."""
    kept: list[ContextItem] = []
    dropped: list[ContextItem] = []
    seen: list[set[str]] = []
    for item in sorted(items, key=lambda it: -it.priority):
        if item.kind != "example":
            kept.append(item)
            continue
        feats = stems(item.text)
        if any(jaccard(feats, other) >= threshold for other in seen):
            dropped.append(item)
            continue
        seen.append(feats)
        kept.append(item)
    return kept, dropped


class ContextAssembler:
    """Fit prioritised items into ``budget`` tokens, compacting first if asked to."""

    def __init__(self, budget: int, dedupe: bool = True, dedupe_threshold: float = 0.9) -> None:
        self.budget = budget
        self.dedupe = dedupe
        self.dedupe_threshold = dedupe_threshold

    def assemble(self, items: Iterable[ContextItem]) -> Assembly:
        items = list(items)
        offered = sum(item.tokens for item in items)
        dropped: list[tuple[ContextItem, str]] = []
        if self.dedupe:
            items, dups = drop_near_duplicates(items, self.dedupe_threshold)
            dropped += [(item, "near-duplicate") for item in dups]
        used = sum(item.tokens for item in items if item.required)
        kept: list[ContextItem] = [item for item in items if item.required]
        for item in sorted((it for it in items if not it.required), key=lambda it: -it.priority):
            if used + item.tokens <= self.budget:
                kept.append(item)
                used += item.tokens
            else:
                dropped.append((item, "over budget"))
        return Assembly(order(kept), arrange(kept), dropped, offered)


def arrange(items: list[ContextItem]) -> list[ContextItem]:
    """Stable system prefix first, examples from least to most relevant, then profile and query last."""
    rank = {"system": 0, "example": 1, "profile": 2, "query": 3}
    return sorted(items, key=lambda it: (rank[it.kind], it.priority if it.kind == "example" else 0))


def order(items: list[ContextItem]) -> list[Message]:
    """Turn arranged items into chat messages: one system message, example pairs, one final user message."""
    items = arrange(items)
    messages = [Message("system", "\n".join(it.text for it in items if it.kind == "system"))]
    for ex in (it for it in items if it.kind == "example"):
        messages += [Message("user", ex.text), Message("assistant", ex.answer)]
    messages.append(Message("user", "\n".join(it.text for it in items if it.kind in ("profile", "query"))))
    return messages


def routing_items(ticket: Ticket, selector: ExampleSelector, cfg: PromptConfig, candidates: int = 16) -> list[ContextItem]:
    """Candidate context for one ticket: system prompt, up to ``candidates`` examples, profile, query."""
    items = [ContextItem("system", system_prompt(cfg), priority=100, required=True, stable=True)]
    q = stems(user_message(ticket, cfg.with_(user_context=False)))
    for ex in selector.select(ticket, candidates, "similar"):
        text = user_message(ex, cfg.with_(user_context=False))
        items.append(ContextItem("example", text, priority=round(jaccard(q, stems(text)), 4),
                                 answer=answer_message(ex.intent, cfg)))
    profile = f"Customer context: language={ticket.language}; product={ticket.product}; channel={ticket.channel}"
    items.append(ContextItem("profile", profile, priority=50))
    items.append(ContextItem("query", user_message(ticket, cfg.with_(user_context=False)), priority=1000,
                             required=True))
    return items


# --------------------------------------------------------------------------- #
# Prefix caching                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class PrefixCache:
    """Simulates a provider-side cache keyed by the exact text of the first message."""

    seen: set[str] = field(default_factory=set)
    hits: int = 0
    misses: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    def observe(self, messages: list[Message]) -> None:
        prefix = messages[0].content
        key = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        self.total_tokens += sum(count_tokens(m.content) + MESSAGE_OVERHEAD for m in messages)
        if key in self.seen:
            self.hits += 1
            self.cached_tokens += count_tokens(prefix) + MESSAGE_OVERHEAD
        else:
            self.misses += 1
            self.seen.add(key)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total else 0.0

    @property
    def cached_share(self) -> float:
        return round(self.cached_tokens / self.total_tokens, 3) if self.total_tokens else 0.0
