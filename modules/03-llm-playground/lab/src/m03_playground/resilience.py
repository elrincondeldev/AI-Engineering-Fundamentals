"""Timeouts, retries with exponential backoff and jitter, and fallback between providers.

Everything here takes a *clock* so the same code runs against real providers
(``RealClock`` really sleeps) and in tests and demos (``VirtualClock`` only adds
numbers, so a "10 second" retry storm finishes in microseconds and always gives
the same result).
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import ErrorKind, PlaygroundError, classify_error
from .metrics import percentile
from .models import SimulatedModel
from .providers import Completion, LLMProvider, Message


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...
    def advance(self, seconds: float) -> None: ...


class RealClock:
    """Wall-clock time. ``advance`` does nothing because real calls already took real time."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def advance(self, seconds: float) -> None:
        return None


class VirtualClock:
    """Simulated time: sleeping and advancing just add to a counter."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


@dataclass(frozen=True)
class RetryPolicy:
    """How hard the gateway tries before giving up on one provider."""

    max_retries: int = 3  # extra attempts after the first one
    base_delay_s: float = 0.5  # first backoff ceiling
    max_delay_s: float = 8.0  # backoff ceiling never grows past this
    jitter: bool = True  # "full jitter": wait a random time between 0 and the ceiling
    timeout_s: float = 10.0  # per-attempt timeout
    deadline_s: float = 30.0  # total budget for the request, waits included


def backoff_delay(retry: int, policy: RetryPolicy, rng: random.Random) -> float:
    """Seconds to wait before retry number ``retry`` (1, 2, 3...).

    The ceiling doubles each time (0.5, 1, 2, 4... capped at ``max_delay_s``). With
    jitter, the wait is uniform between 0 and the ceiling, which spreads out clients
    that all failed at the same moment.
    """
    ceiling = min(policy.max_delay_s, policy.base_delay_s * 2 ** (retry - 1))
    return rng.uniform(0, ceiling) if policy.jitter else ceiling


class FlakyProvider:
    """Wraps a provider and fails on purpose, deterministically.

    Give either a ``script`` (one entry per call: an :class:`ErrorKind` or ``None`` for
    success) or a ``failure_rate`` with a ``seed``.
    """

    def __init__(
        self,
        inner: LLMProvider,
        *,
        script: Sequence[ErrorKind | None] | None = None,
        failure_rate: float = 0.0,
        kinds: Sequence[ErrorKind] = (ErrorKind.SERVER, ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT),
        seed: int = 0,
        retry_after_s: float = 1.0,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.script = list(script) if script is not None else None
        self.failure_rate = failure_rate
        self.kinds = tuple(kinds)
        self.retry_after_s = retry_after_s
        self._rng = random.Random(seed)
        self.calls = 0

    def _next_failure(self) -> ErrorKind | None:
        index = self.calls
        self.calls += 1
        if self.script is not None:
            return self.script[index] if index < len(self.script) else None
        if self._rng.random() < self.failure_rate:
            return self._rng.choice(self.kinds)
        return None

    def complete(self, messages: Sequence[Message], **params: Any) -> Completion:
        kind = self._next_failure()
        if kind is not None:
            retry_after = self.retry_after_s if kind is ErrorKind.RATE_LIMIT else None
            raise PlaygroundError(kind, f"simulated failure from {self.model}", retry_after_s=retry_after)
        return self.inner.complete(messages, **params)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.inner.embed(texts)


@dataclass
class Attempt:
    """One call to one provider, as recorded in the trace."""

    provider: str
    model: str
    number: int  # 1-based attempt number for this provider
    outcome: str  # "ok" or an ErrorKind value
    waited_s: float  # backoff wait before this attempt


@dataclass
class GatewayResult:
    completion: Completion
    attempts: list[Attempt]
    elapsed_s: float  # including waits and timeouts

    @property
    def fallback_used(self) -> bool:
        return self.attempts[-1].model != self.attempts[0].model or (
            self.attempts[-1].provider != self.attempts[0].provider
        )


class GatewayError(RuntimeError):
    """Raised when every provider failed, or when the request itself is wrong."""

    def __init__(self, kind: ErrorKind, attempts: list[Attempt], message: str) -> None:
        super().__init__(message if message.startswith(kind.value) else f"{kind.value}: {message}")
        self.kind = kind
        self.attempts = attempts


@dataclass
class ModelGateway:
    """Calls providers in order: retries transient errors, falls back, respects a deadline."""

    providers: Sequence[LLMProvider]
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    clock: Clock = field(default_factory=VirtualClock)
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("the gateway needs at least one provider")
        self._rng = random.Random(self.seed)

    def complete(self, messages: Sequence[Message], **params: Any) -> GatewayResult:
        start = self.clock.now()
        attempts: list[Attempt] = []
        last_kind = ErrorKind.UNKNOWN
        for provider in self.providers:
            retry, waited = 0, 0.0
            while True:
                remaining = self.policy.deadline_s - (self.clock.now() - start)
                if remaining <= 0:
                    raise GatewayError(ErrorKind.TIMEOUT, attempts, "deadline exceeded")
                try:
                    completion = provider.complete(messages, **params)
                except Exception as exc:
                    kind = last_kind = classify_error(exc)
                    if kind is ErrorKind.TIMEOUT:  # we waited this long for nothing
                        self.clock.advance(min(self.policy.timeout_s, remaining))
                    attempts.append(Attempt(provider.name, provider.model, retry + 1, kind.value, waited))
                    if not kind.fallback:  # the request is wrong: retrying or switching cannot help
                        raise GatewayError(kind, attempts, str(exc)) from exc
                    if not kind.retryable or retry >= self.policy.max_retries:
                        break  # try the next provider
                    retry += 1
                    waited = backoff_delay(retry, self.policy, self._rng)
                    if isinstance(exc, PlaygroundError) and exc.retry_after_s:
                        waited = max(waited, exc.retry_after_s)  # the server told us when to come back
                    if self.clock.now() - start + waited > self.policy.deadline_s:
                        raise GatewayError(ErrorKind.TIMEOUT, attempts, "deadline exceeded") from exc
                    self.clock.sleep(waited)
                    continue
                self.clock.advance(completion.latency_ms / 1000)
                attempts.append(Attempt(provider.name, provider.model, retry + 1, "ok", waited))
                return GatewayResult(completion, attempts, self.clock.now() - start)
        raise GatewayError(last_kind, attempts, "all providers failed")


# --------------------------------------------------------------------------- #
# Reliability experiment (``uv run lab retry``)                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReliabilityRow:
    strategy: str
    success_rate: float
    mean_attempts: float
    p50_s: float
    p95_s: float


def simulate_reliability(
    strategy: str,
    *,
    requests: int = 1000,
    primary_failure_rate: float = 0.3,
    secondary_failure_rate: float = 0.05,
    max_retries: int = 3,
    fallback: bool = False,
    seed: int = 7,
) -> ReliabilityRow:
    """Send ``requests`` independent requests through a gateway and summarise what happened."""
    question = [Message("user", "How long is the warranty?")]
    primary = FlakyProvider(SimulatedModel("fake-small"), failure_rate=primary_failure_rate, seed=seed)
    chain: list[LLMProvider] = [primary]
    if fallback:
        chain.append(
            FlakyProvider(SimulatedModel("fake-large"), failure_rate=secondary_failure_rate, seed=seed + 1)
        )
    gateway = ModelGateway(chain, RetryPolicy(max_retries=max_retries), VirtualClock(), seed=seed)
    ok, attempts, latencies = 0, 0, []
    for _ in range(requests):
        before = gateway.clock.now()
        try:
            result = gateway.complete(question, max_tokens=64)
            ok += 1
            attempts += len(result.attempts)
        except GatewayError as err:
            attempts += len(err.attempts)
        latencies.append(gateway.clock.now() - before)
    return ReliabilityRow(
        strategy=strategy,
        success_rate=ok / requests,
        mean_attempts=attempts / requests,
        p50_s=percentile(latencies, 50),
        p95_s=percentile(latencies, 95),
    )


def reliability_table(requests: int = 1000, seed: int = 7) -> list[ReliabilityRow]:
    """The five scenarios discussed in lesson 3.2."""
    return [
        simulate_reliability("no retries", requests=requests, max_retries=0, seed=seed),
        simulate_reliability("3 retries", requests=requests, max_retries=3, seed=seed),
        simulate_reliability("3 retries + fallback", requests=requests, fallback=True, seed=seed),
        simulate_reliability("outage: 3 retries", requests=requests, primary_failure_rate=1.0, seed=seed),
        simulate_reliability(
            "outage: 3 retries + fallback",
            requests=requests,
            primary_failure_rate=1.0,
            fallback=True,
            seed=seed,
        ),
    ]
