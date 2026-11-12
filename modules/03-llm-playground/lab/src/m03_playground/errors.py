"""Error taxonomy for model calls: what went wrong, and is it worth trying again?

Every back-end fails in its own words (HTTP status codes, socket errors,
exception classes). The gateway only needs to answer three questions, so we
map every failure to one :class:`ErrorKind` first:

1. Should I retry the same provider?  (``kind.retryable``)
2. Should I try another provider?     (``kind.fallback``)
3. Should I stop and fix the request? (neither)
"""

from __future__ import annotations

import re
from enum import Enum

from .providers import ProviderError


class ErrorKind(Enum):
    """The failure classes the playground distinguishes."""

    TIMEOUT = "timeout"  # no answer in time; the server may be overloaded
    RATE_LIMIT = "rate_limit"  # HTTP 429: too many requests or tokens per minute
    SERVER = "server"  # HTTP 5xx: the provider broke, not you
    NETWORK = "network"  # connection refused, DNS failure, reset
    AUTH = "auth"  # HTTP 401/403: wrong or missing key, no access to the model
    BAD_REQUEST = "bad_request"  # HTTP 400/404/422: your request is wrong
    CONTEXT_LENGTH = "context_length"  # prompt + max_tokens does not fit the context window
    UNKNOWN = "unknown"

    @property
    def retryable(self) -> bool:
        """Transient failures: the same request may succeed a moment later."""
        return self in {ErrorKind.TIMEOUT, ErrorKind.RATE_LIMIT, ErrorKind.SERVER, ErrorKind.NETWORK}

    @property
    def fallback(self) -> bool:
        """Failures where another provider could still answer the same request."""
        return self.retryable or self is ErrorKind.AUTH


class PlaygroundError(RuntimeError):
    """A classified failure. ``retry_after_s`` is set for rate limits that say when to come back."""

    def __init__(self, kind: ErrorKind, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(f"{kind.value}: {message}")
        self.kind = kind
        self.retry_after_s = retry_after_s


_HTTP_RE = re.compile(r"HTTP (\d{3})")


def classify_error(exc: BaseException) -> ErrorKind:
    """Map any exception raised by a provider to an :class:`ErrorKind`."""
    if isinstance(exc, PlaygroundError):
        return exc.kind
    if isinstance(exc, TimeoutError):
        return ErrorKind.TIMEOUT
    if isinstance(exc, ProviderError):
        text = str(exc)
        match = _HTTP_RE.search(text)
        if match:
            status = int(match.group(1))
            if status == 429:
                return ErrorKind.RATE_LIMIT
            if status in (401, 403):
                return ErrorKind.AUTH
            if status >= 500:
                return ErrorKind.SERVER
            if "context" in text.lower() and ("length" in text.lower() or "window" in text.lower()):
                return ErrorKind.CONTEXT_LENGTH
            return ErrorKind.BAD_REQUEST
        if "timed out" in text.lower():
            return ErrorKind.TIMEOUT
        if text.startswith("Cannot reach"):
            return ErrorKind.NETWORK
    if isinstance(exc, ConnectionError):
        return ErrorKind.NETWORK
    return ErrorKind.UNKNOWN
