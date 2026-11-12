"""Simulated chat models with a virtual clock, built on the shared ``FakeProvider``.

The playground must print numbers that are identical on every machine: time to
first token, throughput, cost. A real model cannot give you that, so the
default back-end is a *simulated* model:

- the text comes from ``FakeProvider`` with a ``responder`` (as in every lab);
- the timing comes from a small latency model (overhead + prefill + decode) that is
  *computed*, not measured, so it never varies between runs;
- temperature and seed change which canned answer is sampled, so parameter sweeps
  behave like a (very small) real model, including the occasional wrong answer.

Every profile value below is a fictional teaching number, not a measurement of a real model.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .errors import ErrorKind, PlaygroundError
from .providers import Completion, FakeProvider, Message, count_tokens

_CHUNK_RE = re.compile(r"\s*(?:\w+|[^\w\s])", re.UNICODE)


@dataclass(frozen=True)
class ModelProfile:
    """How a simulated model behaves. All numbers are invented for teaching."""

    name: str
    context_window: int  # max prompt tokens + max_tokens per request
    overhead_ms: float  # network, queueing and scheduling before any work starts
    prefill_ms_per_token: float  # time to read each prompt token
    decode_ms_per_token: float  # time to generate each output token after the first
    logit_scale: float  # > 1 makes the model more confident (sharper sampling)


PROFILES: dict[str, ModelProfile] = {
    "fake-small": ModelProfile("fake-small", 1024, 120.0, 0.25, 20.0, 1.0),
    "fake-large": ModelProfile("fake-large", 8192, 300.0, 0.60, 45.0, 2.0),
}


@dataclass(frozen=True)
class Timing:
    """Simulated timing of one request, in milliseconds since the request was sent."""

    ttft_ms: float  # time to first token
    total_ms: float  # time until the last token arrived

    def token_times(self, n_tokens: int) -> list[float]:
        """Arrival time of each output token (the first one arrives at ``ttft_ms``)."""
        if n_tokens <= 0:
            return []
        if n_tokens == 1:
            return [self.ttft_ms]
        step = (self.total_ms - self.ttft_ms) / (n_tokens - 1)
        return [self.ttft_ms + i * step for i in range(n_tokens)]


def simulate_timing(profile: ModelProfile, prompt_tokens: int, completion_tokens: int) -> Timing:
    """Latency model: TTFT = overhead + prefill x prompt tokens; then one decode step per extra token."""
    ttft = profile.overhead_ms + profile.prefill_ms_per_token * prompt_tokens
    total = ttft + profile.decode_ms_per_token * max(completion_tokens - 1, 0)
    return Timing(ttft_ms=round(ttft, 3), total_ms=round(total, 3))


# --------------------------------------------------------------------------- #
# What the simulated model "knows": a few Hearthwise facts (from datasets/)    #
# --------------------------------------------------------------------------- #

# (keywords, [(candidate answer, score)]). Higher score = more likely. A few
# low-score candidates are wrong on purpose: high temperatures surface them.
TOPICS: list[tuple[tuple[str, ...], list[tuple[str, float]]]] = [
    (
        ("return", "refund", "send back", "devol"),
        [
            (
                "You can return any Hearthwise device within 30 days of delivery, opened or not. There is no "
                "restocking fee, and the refund arrives 5 to 7 business days after we receive the item.",
                2.0,
            ),
            (
                "Hearthwise accepts returns for 30 days after delivery with no restocking fee. "
                "Start the return in the app under Account > Orders.",
                1.6,
            ),
            ("Returns are accepted within 14 days, and opened items pay a 10% restocking fee.", 0.4),
        ],
    ),
    (
        ("warranty", "garant", "faulty", "broken", "stopped working"),
        [
            (
                "Every Hearthwise device has a 2-year warranty from the delivery date. With an active "
                "Hearthwise Plus subscription it is 3 years.",
                2.0,
            ),
            (
                "The warranty lasts 2 years, plus one extra year for Hearthwise Plus members. "
                "Open a claim in the app under Support > Warranty claim.",
                1.6,
            ),
            ("The warranty lasts 1 year.", 0.2),
        ],
    ),
    (
        ("shipping", "delivery", "deliver", "arrive", "envío", "envio", "tracking", "where is my order"),
        [
            (
                "Standard delivery in Spain and Portugal takes 2 to 4 business days and costs €4.95; "
                "it is free for orders over €49.",
                2.0,
            ),
            ("Express delivery takes 1 business day and costs €6.95 if you order before 14:00.", 1.4),
            ("Delivery to the rest of the European Union takes 3 to 6 business days and costs €7.95.", 1.2),
        ],
    ),
    (
        ("hub", "homemesh", "how many devices"),
        [
            ("The Hearth Hub connects up to 64 HomeMesh devices and runs automations locally.", 2.0),
            ("One Hearth Hub supports up to 64 HomeMesh devices, such as Watcher S1 sensors.", 1.6),
            ("The Hearth Hub supports up to 250 devices.", 0.3),
        ],
    ),
    (
        ("plus", "subscription", "suscrip"),
        [
            ("Hearthwise Plus costs €4.99 per month or €49 per year.", 2.0),
            ("Hearthwise Plus is €49 per year, or €4.99 if you pay monthly.", 1.6),
        ],
    ),
]

GENERIC: list[tuple[str, float]] = [
    (
        "I am the offline simulated model, so I only know a few Hearthwise topics: returns, warranty, "
        'shipping, the Hearth Hub and Hearthwise Plus. You said: "{preview}"',
        1.0,
    ),
    (
        "Simulated reply. Ask me about returns, warranty, shipping, the Hearth Hub or Hearthwise Plus. "
        'Your message was: "{preview}"',
        0.8,
    ),
]


def candidates_for(user_text: str) -> list[tuple[str, float]]:
    """Return the candidate answers for a user message (first matching topic wins)."""
    low = user_text.lower()
    for keywords, cands in TOPICS:
        if any(k in low for k in keywords):
            return cands
    preview = " ".join(user_text.split())[:60]
    return [(text.replace("{preview}", preview), score) for text, score in GENERIC]


def sample_index(scores: Sequence[float], temperature: float, rng: random.Random) -> int:
    """Pick a candidate: greedy at temperature 0, otherwise softmax(score / temperature)."""
    if temperature <= 0:
        return max(range(len(scores)), key=lambda i: scores[i])
    top = max(scores)
    weights = [math.exp((s - top) / temperature) for s in scores]
    return rng.choices(range(len(scores)), weights=weights, k=1)[0]


def _digest(messages: Sequence[Message]) -> str:
    h = hashlib.sha256()
    for m in messages:
        h.update(f"{m.role}:{m.content}\x00".encode())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class StreamChunk:
    """One piece of a streamed answer and when it arrived (simulated milliseconds)."""

    text: str
    t_ms: float


def chunks_from_completion(completion: Completion) -> Iterator[StreamChunk]:
    """Split a simulated completion into token-sized chunks with their arrival times."""
    pieces = _CHUNK_RE.findall(completion.text)
    ttft = completion.raw.get("ttft_ms", completion.latency_ms)
    for piece, t in zip(pieces, Timing(ttft, completion.latency_ms).token_times(len(pieces))):
        yield StreamChunk(piece, t)


class SimulatedModel:
    """An offline chat model that satisfies the ``LLMProvider`` protocol from ``providers.py``."""

    name = "fake"

    def __init__(self, profile: ModelProfile | str = "fake-small") -> None:
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.model = self.profile.name
        self._embedder = FakeProvider()

    def reply(self, messages: Sequence[Message], *, temperature: float, seed: int | None) -> str:
        """Choose the answer text. The same messages, temperature and seed give the same text."""
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        cands = candidates_for(user)
        scores = [s * self.profile.logit_scale for _, s in cands]
        rng = random.Random(f"{seed or 0}:{_digest(messages)}")
        text = cands[sample_index(scores, temperature, rng)][0]
        system = " ".join(m.content.lower() for m in messages if m.role == "system")
        if "one sentence" in system:
            text = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
        return text

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Completion:
        prompt = sum(count_tokens(m.content) + 4 for m in messages)
        if prompt + max_tokens > self.profile.context_window:
            raise PlaygroundError(
                ErrorKind.CONTEXT_LENGTH,
                f"{prompt} prompt tokens + max_tokens {max_tokens} > context window "
                f"{self.profile.context_window} of {self.model}",
            )
        text = self.reply(messages, temperature=temperature, seed=seed)
        pieces = _CHUNK_RE.findall(text)
        truncated = len(pieces) > max_tokens  # a real model stops after exactly max_tokens tokens
        if truncated:
            text = "".join(pieces[:max_tokens])
        # Reuse FakeProvider for stop sequences and token accounting.
        base = FakeProvider(responder=lambda _msgs: text, model=self.model).complete(
            messages, max_tokens=max_tokens, stop=stop
        )
        timing = simulate_timing(self.profile, base.prompt_tokens, base.completion_tokens)
        return replace(
            base,
            latency_ms=timing.total_ms,
            finish_reason="length" if truncated else base.finish_reason,
            raw={"ttft_ms": timing.ttft_ms, "simulated": True},
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> Iterator[StreamChunk]:
        """Yield the answer token by token with simulated arrival times (no real waiting)."""
        done = self.complete(messages, temperature=temperature, max_tokens=max_tokens, seed=seed)
        yield from chunks_from_completion(done)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embedder.embed(texts)
