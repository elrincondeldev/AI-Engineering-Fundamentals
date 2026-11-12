"""Error taxonomy, backoff, retries, fallback and deadlines (all on a virtual clock)."""

from __future__ import annotations

import random

import pytest

from m03_playground.errors import ErrorKind, PlaygroundError, classify_error
from m03_playground.models import SimulatedModel
from m03_playground.providers import Message, ProviderError
from m03_playground.resilience import (
    FlakyProvider,
    GatewayError,
    ModelGateway,
    RetryPolicy,
    VirtualClock,
    backoff_delay,
)

Q = [Message("user", "How long is the warranty?")]


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("HTTP 429 from http://x: slow down", ErrorKind.RATE_LIMIT),
        ("HTTP 503 from http://x: overloaded", ErrorKind.SERVER),
        ("HTTP 401 from http://x: bad key", ErrorKind.AUTH),
        ("HTTP 400 from http://x: maximum context length exceeded", ErrorKind.CONTEXT_LENGTH),
        ("HTTP 400 from http://x: unknown field", ErrorKind.BAD_REQUEST),
        ("Cannot reach http://x: timed out", ErrorKind.TIMEOUT),
        ("Cannot reach http://x: connection refused", ErrorKind.NETWORK),
    ],
)
def test_classify_provider_errors(message: str, kind: ErrorKind) -> None:
    assert classify_error(ProviderError(message)) is kind


def test_retryable_and_fallback_flags() -> None:
    assert ErrorKind.RATE_LIMIT.retryable and ErrorKind.TIMEOUT.retryable
    assert not ErrorKind.BAD_REQUEST.retryable and not ErrorKind.BAD_REQUEST.fallback
    assert not ErrorKind.AUTH.retryable and ErrorKind.AUTH.fallback
    assert not ErrorKind.CONTEXT_LENGTH.fallback
    assert classify_error(ValueError("x")) is ErrorKind.UNKNOWN


def test_backoff_ceiling_doubles_and_caps() -> None:
    policy = RetryPolicy(jitter=False)
    rng = random.Random(0)
    assert [backoff_delay(n, policy, rng) for n in range(1, 7)] == [0.5, 1, 2, 4, 8, 8]


def test_full_jitter_stays_under_the_ceiling() -> None:
    policy = RetryPolicy()
    rng = random.Random(1)
    for n in range(1, 6):
        ceiling = min(8.0, 0.5 * 2 ** (n - 1))
        assert all(0 <= backoff_delay(n, policy, rng) <= ceiling for _ in range(200))


def test_retries_until_success() -> None:
    flaky = FlakyProvider(SimulatedModel(), script=[ErrorKind.SERVER, ErrorKind.RATE_LIMIT, None])
    result = ModelGateway([flaky]).complete(Q)
    assert [a.outcome for a in result.attempts] == ["server", "rate_limit", "ok"]
    assert result.attempts[2].waited_s >= 1.0  # honours retry_after of the rate limit
    assert not result.fallback_used


def test_falls_back_after_retries_are_exhausted() -> None:
    flaky = FlakyProvider(SimulatedModel("fake-small"), script=[ErrorKind.SERVER] * 4)
    result = ModelGateway([flaky, SimulatedModel("fake-large")]).complete(Q)
    assert [a.outcome for a in result.attempts] == ["server"] * 4 + ["ok"]
    assert result.completion.model == "fake-large"
    assert result.fallback_used


def test_auth_errors_skip_retries_but_fall_back() -> None:
    flaky = FlakyProvider(SimulatedModel("fake-small"), script=[ErrorKind.AUTH])
    result = ModelGateway([flaky, SimulatedModel("fake-large")]).complete(Q)
    assert [a.outcome for a in result.attempts] == ["auth", "ok"]


@pytest.mark.parametrize("kind", [ErrorKind.BAD_REQUEST, ErrorKind.CONTEXT_LENGTH])
def test_bad_requests_fail_fast(kind: ErrorKind) -> None:
    flaky = FlakyProvider(SimulatedModel(), script=[kind])
    with pytest.raises(GatewayError) as err:
        ModelGateway([flaky, SimulatedModel("fake-large")]).complete(Q)
    assert err.value.kind is kind
    assert len(err.value.attempts) == 1


def test_deadline_stops_retrying() -> None:
    flaky = FlakyProvider(SimulatedModel(), script=[ErrorKind.TIMEOUT] * 4)
    clock = VirtualClock()
    with pytest.raises(GatewayError) as err:
        ModelGateway(
            [flaky, SimulatedModel("fake-large")], RetryPolicy(timeout_s=10, deadline_s=30), clock
        ).complete(Q)
    assert err.value.kind is ErrorKind.TIMEOUT
    assert clock.now() <= 30.0 + 1e-9


def test_shorter_timeout_leaves_room_for_the_fallback() -> None:
    flaky = FlakyProvider(SimulatedModel(), script=[ErrorKind.TIMEOUT] * 4)
    result = ModelGateway([flaky, SimulatedModel("fake-large")], RetryPolicy(timeout_s=2)).complete(Q)
    assert result.completion.model == "fake-large"


def test_flaky_failure_rate_is_seeded() -> None:
    def outcomes(seed: int) -> list[bool]:
        flaky = FlakyProvider(SimulatedModel(), failure_rate=0.5, seed=seed)
        out = []
        for _ in range(20):
            try:
                flaky.complete(Q)
                out.append(True)
            except PlaygroundError:
                out.append(False)
        return out

    assert outcomes(3) == outcomes(3)
    assert 0 < sum(outcomes(3)) < 20
