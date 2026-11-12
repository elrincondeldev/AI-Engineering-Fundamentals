"""A small interpolated trigram language model.

It plays the role of a base model: given a context it returns one score (a log-probability)
per vocabulary entry, exactly the vector of logits a transformer produces. Every decoding
strategy in :mod:`decoding` works on that vector, so it would work unchanged on a real model.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

BOS, EOS, UNK = "<bos>", "<eos>", "<unk>"
SPECIAL = (BOS, EOS, UNK)


class NgramLM:
    """Interpolated trigram model: P = l3*P(w|u,v) + l2*P(w|v) + l1*P(w).

    Words seen fewer than ``min_count`` times become ``<unk>``. Each document is wrapped as
    ``<bos> <bos> tokens... <eos>`` so the model can learn how documents end.
    """

    def __init__(self, lambdas: tuple[float, float, float] = (0.1, 0.3, 0.6), min_count: int = 2) -> None:
        if not math.isclose(sum(lambdas), 1.0):
            raise ValueError("lambdas must sum to 1")
        self.lambdas = lambdas
        self.min_count = min_count
        self.vocab: list[str] = []
        self.index: dict[str, int] = {}
        self._uni = np.zeros(0)
        self._bi: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._tri: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self.n_train_tokens = 0

    # ------------------------------------------------------------------ training
    def fit(self, docs: Iterable[Sequence[str]]) -> NgramLM:
        docs = [list(d) for d in docs]
        counts = Counter(t for d in docs for t in d)
        kept = sorted(t for t, c in counts.items() if c >= self.min_count)
        self.vocab = list(SPECIAL) + kept
        self.index = {t: i for i, t in enumerate(self.vocab)}
        uni: Counter[str] = Counter()
        bi: dict[str, Counter[str]] = defaultdict(Counter)
        tri: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for doc in docs:
            seq = [BOS, BOS] + [self.map(t) for t in doc] + [EOS]
            for i in range(2, len(seq)):
                u, v, w = seq[i - 2], seq[i - 1], seq[i]
                uni[w] += 1
                bi[v][w] += 1
                tri[(u, v)][w] += 1
        self.n_train_tokens = sum(uni.values())
        vec = np.ones(len(self.vocab))  # add-one smoothing keeps every word possible
        for w, c in uni.items():
            vec[self.index[w]] += c
        vec[self.index[BOS]] = 0.0
        self._uni = vec / vec.sum()
        self._bi = {v: self._pack(c) for v, c in bi.items()}
        self._tri = {k: self._pack(c) for k, c in tri.items()}
        return self

    def _pack(self, counter: Counter[str]) -> tuple[np.ndarray, np.ndarray]:
        idx = np.array([self.index[w] for w in counter], dtype=np.int64)
        cnt = np.array(list(counter.values()), dtype=np.float64)
        return idx, cnt / cnt.sum()

    def map(self, token: str) -> str:
        """Map a token to itself if known, else to ``<unk>``."""
        return token if token in self.index else UNK

    # ------------------------------------------------------------------ inference
    def next_logprobs(self, context: Sequence[str]) -> np.ndarray:
        """Log-probabilities of every vocabulary entry after ``context`` (the model's logits)."""
        ctx = [BOS, BOS] + [self.map(t) for t in context]
        u, v = ctx[-2], ctx[-1]
        l1, l2, l3 = self.lambdas
        probs = l1 * self._uni
        weight_left = l2 + l3
        if v in self._bi:
            idx, p = self._bi[v]
            probs = probs.copy()
            probs[idx] += l2 * p
            weight_left -= l2
        if (u, v) in self._tri:
            idx, p = self._tri[(u, v)]
            probs = probs.copy()
            probs[idx] += l3 * p
            weight_left -= l3
        if weight_left > 0:  # unseen contexts give their weight back to the lower orders
            probs = probs / probs.sum()
        with np.errstate(divide="ignore"):
            return np.log(probs)

    def sequence_logprob(self, tokens: Sequence[str], context: Sequence[str] = ()) -> float:
        """Sum of log P(token | previous tokens) over ``tokens`` given an optional ``context``."""
        history = list(context)
        total = 0.0
        for tok in tokens:
            total += float(self.next_logprobs(history)[self.index[self.map(tok)]])
            history.append(tok)
        return total

    def perplexity(self, docs: Iterable[Sequence[str]]) -> float:
        """exp(average negative log-likelihood per token), counting each document's ``<eos>``."""
        nll, n = 0.0, 0
        for doc in docs:
            tokens = list(doc) + [EOS]
            nll -= self.sequence_logprob(tokens)
            n += len(tokens)
        return math.exp(nll / n)
