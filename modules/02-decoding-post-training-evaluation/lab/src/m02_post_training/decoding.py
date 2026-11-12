"""Decoding strategies: turn a vector of logits into text.

Every function here takes plain numpy arrays, so the same code works with any model that
returns one logit per vocabulary entry. The n-gram model in :mod:`ngram` is the stand-in.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .ngram import BOS, EOS, UNK
from .text import detokenize, ngrams, tokenize

Strategy = Literal["greedy", "beam", "sample"]
NEG_INF = -np.inf


class LogitModel(Protocol):
    """Anything that maps a context to one logit per vocabulary entry."""

    vocab: list[str]
    index: dict[str, int]

    def next_logprobs(self, context: Sequence[str]) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Building blocks                                                             #
# --------------------------------------------------------------------------- #


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Probabilities from logits. Temperature divides the logits before normalising."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use greedy decoding for temperature 0)")
    z = logits / temperature
    z = z - np.max(z)  # numerical stability: the largest logit becomes 0
    e = np.exp(z)
    return e / e.sum()


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` highest logits; set the rest to -inf."""
    if k <= 0 or k >= logits.size:
        return logits
    cutoff = np.partition(logits, -k)[-k]
    return np.where(logits >= cutoff, logits, NEG_INF)


def top_p_filter(logits: np.ndarray, p: float, temperature: float = 1.0) -> np.ndarray:
    """Nucleus filter: keep the smallest set of tokens whose probabilities sum to at least ``p``."""
    if p >= 1.0:
        return logits
    probs = softmax(logits, temperature)
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    n_keep = int(np.searchsorted(cumulative, p - 1e-9) + 1)  # first position where the sum reaches p
    keep = order[:n_keep]
    out = np.full_like(logits, NEG_INF)
    out[keep] = logits[keep]
    return out


def apply_repetition_penalty(logits: np.ndarray, history: Sequence[int], penalty: float) -> np.ndarray:
    """Make tokens that already appeared less likely (the multiplicative rule used by many libraries).

    Positive logits are divided by ``penalty`` and negative ones multiplied by it, so the
    token always moves down. ``penalty = 1`` changes nothing.
    """
    if penalty == 1.0 or not history:
        return logits
    out = logits.copy()
    seen = np.unique(np.asarray(history, dtype=np.int64))
    vals = out[seen]
    out[seen] = np.where(vals > 0, vals / penalty, vals * penalty)
    return out


def apply_frequency_penalty(
    logits: np.ndarray, history: Sequence[int], frequency: float = 0.0, presence: float = 0.0
) -> np.ndarray:
    """Additive penalties used by OpenAI-style APIs: subtract ``frequency * count + presence``."""
    if not history or (frequency == 0.0 and presence == 0.0):
        return logits
    out = logits.copy()
    for tok, count in Counter(history).items():
        out[tok] -= frequency * count + presence
    return out


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DecodeConfig:
    """All the knobs a generation API usually exposes."""

    strategy: Strategy = "sample"
    temperature: float = 1.0
    top_k: int = 0  # 0 = off
    top_p: float = 1.0  # 1.0 = off
    repetition_penalty: float = 1.0  # 1.0 = off
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    num_beams: int = 4
    max_new_tokens: int = 40
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass
class Generation:
    """The result of one decoding run."""

    tokens: list[str]
    text: str
    finish_reason: str  # "stop" (a stop sequence), "eos" (model ended) or "length" (max tokens)
    logprob: float = 0.0  # sum of the model's log-probabilities of the generated tokens


def _banned(model: LogitModel) -> list[int]:
    return [model.index[t] for t in (BOS, UNK) if t in model.index]


def next_logits(model: LogitModel, context: Sequence[str]) -> np.ndarray:
    """The model's log-probabilities with ``<bos>`` and ``<unk>`` removed and the rest renormalised.

    A generator must never emit those two tokens, so every strategy starts from this vector.
    """
    logits = model.next_logprobs(context).copy()
    logits[_banned(model)] = NEG_INF
    return logits - np.logaddexp.reduce(logits[np.isfinite(logits)])


def _cut_at_stop(tokens: list[str], stop: Sequence[str]) -> tuple[list[str], bool]:
    """If the generated text contains a stop sequence, drop the tokens from it onwards."""
    if not stop:
        return tokens, False
    text = detokenize(tokens)
    hits = [text.find(s) for s in stop if s and s in text]
    if not hits:
        return tokens, False
    cut = min(hits)
    kept: list[str] = []
    for i in range(1, len(tokens) + 1):
        if len(detokenize(tokens[:i])) > cut:
            break
        kept = tokens[:i]
    return kept, True


def generate(model: LogitModel, prompt: str, config: DecodeConfig = DecodeConfig()) -> Generation:
    """Generate a continuation of ``prompt`` with greedy, beam or sampling-based decoding."""
    if config.strategy == "beam":
        return beam_search(model, prompt, config)
    rng = np.random.default_rng(config.seed)
    context = tokenize(prompt)
    out: list[str] = []
    history: list[int] = []
    total = 0.0
    eos = model.index[EOS]
    for _ in range(config.max_new_tokens):
        raw = next_logits(model, context + out)
        logits = apply_repetition_penalty(raw, history, config.repetition_penalty)
        logits = apply_frequency_penalty(logits, history, config.frequency_penalty, config.presence_penalty)
        if config.strategy == "greedy":
            tok = int(np.argmax(logits))
        else:
            logits = top_k_filter(logits, config.top_k)
            logits = top_p_filter(logits, config.top_p, config.temperature)
            probs = softmax(logits, config.temperature)
            tok = int(rng.choice(probs.size, p=probs))
        total += float(raw[tok])
        if tok == eos:
            return Generation(out, detokenize(out), "eos", total)
        out.append(model.vocab[tok])
        history.append(tok)
        cut, stopped = _cut_at_stop(out, config.stop)
        if stopped:
            return Generation(cut, detokenize(cut), "stop", total)
    return Generation(out, detokenize(out), "length", total)


def beam_search(model: LogitModel, prompt: str, config: DecodeConfig = DecodeConfig()) -> Generation:
    """Keep the ``num_beams`` most likely partial sequences at every step; return the best one.

    Scores are summed log-probabilities. Finished beams (that produced ``<eos>``) are set aside;
    the search stops when ``max_new_tokens`` is reached or every beam has finished.
    """
    context = tokenize(prompt)
    eos = model.index[EOS]
    beams: list[tuple[float, list[int]]] = [(0.0, [])]
    finished: list[tuple[float, list[int]]] = []
    for _ in range(config.max_new_tokens):
        candidates: list[tuple[float, list[int]]] = []
        for score, seq in beams:
            logits = next_logits(model, context + [model.vocab[i] for i in seq])
            logits = apply_repetition_penalty(logits, seq, config.repetition_penalty)
            best = np.argsort(-logits, kind="stable")[: config.num_beams]
            candidates.extend((score + float(logits[i]), seq + [int(i)]) for i in best)
        candidates.sort(key=lambda c: -c[0])
        beams = []
        for score, seq in candidates:
            if seq[-1] == eos:
                finished.append((score, seq[:-1]))
            else:
                beams.append((score, seq))
            if len(beams) == config.num_beams:
                break
        if not beams:
            break
    pool = finished + beams
    score, seq = max(pool, key=lambda c: c[0])
    tokens = [model.vocab[i] for i in seq]
    reason = "eos" if (score, seq) in finished else "length"
    cut, stopped = _cut_at_stop(tokens, config.stop)
    return Generation(cut, detokenize(cut), "stop" if stopped else reason, score)


# --------------------------------------------------------------------------- #
# Inspecting a distribution                                                   #
# --------------------------------------------------------------------------- #


def top_tokens(
    model: LogitModel, prompt: str, temperature: float = 1.0, n: int = 5
) -> list[tuple[str, float]]:
    """The ``n`` most likely next tokens after ``prompt`` and their probabilities."""
    probs = softmax(next_logits(model, tokenize(prompt)), temperature)
    order = np.argsort(-probs, kind="stable")[:n]
    return [(model.vocab[i], float(probs[i])) for i in order]


def nucleus_size(model: LogitModel, prompt: str, p: float, temperature: float = 1.0) -> int:
    """How many tokens top-p keeps after ``prompt``: small when the model is confident."""
    logits = next_logits(model, tokenize(prompt))
    return int(np.isfinite(top_p_filter(logits, p, temperature)).sum())


# --------------------------------------------------------------------------- #
# Diversity and repetition                                                    #
# --------------------------------------------------------------------------- #


def distinct_n(outputs: Sequence[Sequence[str]], n: int) -> float:
    """Unique n-grams divided by total n-grams, pooled over all outputs (higher = more varied)."""
    grams = [g for toks in outputs for g in ngrams(list(toks), n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_rate(tokens: Sequence[str], n: int = 3) -> float:
    """Share of n-grams in one output that already appeared earlier in the same output."""
    grams = ngrams(list(tokens), n)
    if not grams:
        return 0.0
    seen: set[tuple[str, ...]] = set()
    repeats = 0
    for g in grams:
        repeats += g in seen
        seen.add(g)
    return repeats / len(grams)


# --------------------------------------------------------------------------- #
# The benchmark used in lesson 2.1                                            #
# --------------------------------------------------------------------------- #

BENCH_PROMPTS = (
    "the ember t3",
    "to reset the",
    "the hearth hub",
    "returns are",
    "the lookout c2",
    "if your bulb",
)

BENCH_STRATEGIES: dict[str, DecodeConfig] = {
    "greedy": DecodeConfig(strategy="greedy"),
    "beam (4 beams)": DecodeConfig(strategy="beam", num_beams=4),
    "sample T=1.0": DecodeConfig(temperature=1.0),
    "sample T=0.7": DecodeConfig(temperature=0.7),
    "sample T=1.5": DecodeConfig(temperature=1.5),
    "top-k k=10": DecodeConfig(top_k=10),
    "top-p p=0.9": DecodeConfig(top_p=0.9),
    "top-p 0.9 + rep 1.3": DecodeConfig(top_p=0.9, repetition_penalty=1.3),
}


@dataclass(frozen=True)
class BenchRow:
    """Diversity and quality summary of one strategy over every prompt and seed."""

    name: str
    distinct_2: float
    rep_3: float
    unique_outputs: int
    n_outputs: int
    logprob_per_token: float
    eos_rate: float


def decoding_benchmark(
    model: LogitModel,
    strategies: dict[str, DecodeConfig] | None = None,
    prompts: Sequence[str] = BENCH_PROMPTS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    max_new_tokens: int = 50,
) -> list[BenchRow]:
    """Run every strategy on every prompt with every seed and summarise the outputs."""
    rows = []
    for name, base in (strategies or BENCH_STRATEGIES).items():
        outs: list[Generation] = []
        for prompt in prompts:
            for seed in seeds:
                cfg = DecodeConfig(**{**base.__dict__, "seed": seed, "max_new_tokens": max_new_tokens})
                outs.append(generate(model, prompt, cfg))
        n_tok = sum(len(g.tokens) for g in outs)
        rows.append(
            BenchRow(
                name=name,
                distinct_2=distinct_n([g.tokens for g in outs], 2),
                rep_3=float(np.mean([repetition_rate(g.tokens, 3) for g in outs])),
                unique_outputs=len({g.text for g in outs}),
                n_outputs=len(outs),
                logprob_per_token=sum(g.logprob for g in outs) / max(n_tok, 1),
                eos_rate=sum(g.finish_reason == "eos" for g in outs) / len(outs),
            )
        )
    return rows
