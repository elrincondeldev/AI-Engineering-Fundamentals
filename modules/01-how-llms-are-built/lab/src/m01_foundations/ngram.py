"""A counting next-word predictor: the simplest possible "language model".

It learns, from the cleaned crawl, how often each word follows each short context, then predicts
the next word by looking up the longest context it has seen ("backoff"). An LLM does the same job,
predicting the next token, but with a neural network that generalises to contexts it never saw.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
DOC_BREAK = "<doc>"


def words(text: str) -> list[str]:
    """Lowercased words and punctuation marks."""
    return [w.lower() for w in _TOKEN_RE.findall(text)]


def corpus_tokens(texts: Sequence[str]) -> list[str]:
    """Concatenate documents into one token stream with a separator between documents."""
    out: list[str] = []
    for t in texts:
        out.extend(words(t))
        out.append(DOC_BREAK)
    return out


@dataclass
class NGramModel:
    """Counts of next words for every context of length 1 .. ``order - 1``."""

    order: int = 4
    counts: dict[tuple[str, ...], Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def fit(self, tokens: Sequence[str]) -> NGramModel:
        for i in range(1, len(tokens)):
            for k in range(1, self.order):
                if i - k < 0:
                    break
                self.counts[tuple(tokens[i - k : i])][tokens[i]] += 1
        return self

    def next_word_probs(self, context: Sequence[str], top: int = 5) -> tuple[int, list[tuple[str, float]]]:
        """Return (context length used, top next words with probabilities), backing off."""
        for k in range(min(self.order - 1, len(context)), 0, -1):
            ctx = tuple(context[-k:])
            if ctx in self.counts:
                c = self.counts[ctx]
                total = sum(c.values())
                ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
                return k, [(w, n / total) for w, n in ranked]
        return 0, []

    def generate(self, prompt: str, max_words: int = 12) -> list[str]:
        """Greedy continuation: always append the single most likely next word."""
        ctx = words(prompt)
        out: list[str] = []
        for _ in range(max_words):
            _, probs = self.next_word_probs(ctx, top=1)
            if not probs or probs[0][0] == DOC_BREAK:
                break
            ctx.append(probs[0][0])
            out.append(probs[0][0])
        return out


def coverage(train: Sequence[str], held_out: Sequence[str], n: int) -> float:
    """Share of held-out n-grams that also appear in the training tokens.

    As n grows, most n-grams have never been seen: counting alone cannot generalise.
    """
    seen = {tuple(train[i : i + n]) for i in range(len(train) - n + 1)}
    grams = [tuple(held_out[i : i + n]) for i in range(len(held_out) - n + 1)]
    grams = [g for g in grams if DOC_BREAK not in g]
    return sum(g in seen for g in grams) / len(grams) if grams else 0.0


def split_docs(texts: Sequence[str], every: int = 5) -> tuple[list[str], list[str]]:
    """Deterministic split: every ``every``-th document is held out."""
    train = [t for i, t in enumerate(texts) if i % every != 0]
    held = [t for i, t in enumerate(texts) if i % every == 0]
    return train, held
