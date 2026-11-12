"""The orchestrator: one request through every stage of the playground.

    1. input guardrails   pseudonymise PII, flag injection attempts, enforce the data boundary
    2. prompt assembly    system prompt + history + new message, cut to the token budget
    3. model gateway      call the provider with retries and fallback
    4. output checks      empty, truncated, PII in the answer, known-wrong claims
    5. store and trace    update the conversation, append one JSONL trace line

Each stage is a small function from another file; this class only wires them together.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .conversation import Conversation, prompt_tokens
from .errors import ErrorKind
from .guardrails import check_output, is_external, scan_injection
from .metrics import EXAMPLE_PRICES, Prices, decode_tokens_per_second, request_cost
from .models import PROFILES, SimulatedModel
from .providers import Completion, LLMProvider, Message, ProviderError, get_provider
from .redact import Pseudonymiser
from .resilience import (
    Attempt,
    Clock,
    FlakyProvider,
    GatewayError,
    ModelGateway,
    RealClock,
    RetryPolicy,
    VirtualClock,
)
from .tracing import TraceWriter, text_hash

DEFAULT_SYSTEM = "You are a helpful assistant for Hearthwise employees. Be accurate and brief."
DEFAULT_CONTEXT_WINDOW = 8192  # assumption for real models; set --context-window to your model's value


class DataBoundaryError(RuntimeError):
    """Raised when a configuration would send unredacted text to an external provider."""


@dataclass
class Settings:
    """Everything a user can change from the CLI or the chat REPL."""

    provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "fake"))
    model: str | None = None  # None: fake-small for the fake provider, the env default otherwise
    fallback_model: str | None = None  # a second model tried when the first one keeps failing
    temperature: float = 0.0
    max_tokens: int = 256
    seed: int | None = 0
    system: str = DEFAULT_SYSTEM
    redact: bool = True
    redactor_level: str = "names"
    policy: str = "pin_system"
    context_window: int | None = None
    prices: Prices | None = None  # None: example prices for simulated models, unknown otherwise
    flaky: Sequence[ErrorKind | None] = ()  # scripted failures for the first provider (demos)
    timeout_s: float = 10.0


def make_provider(provider: str, model: str | None, timeout_s: float = 10.0) -> LLMProvider:
    """Build one back-end. ``fake`` means a simulated model with a virtual clock."""
    if provider == "fake":
        name = model or "fake-small"
        if name not in PROFILES:
            raise ProviderError(f"Unknown simulated model {name!r}. Use one of: {', '.join(PROFILES)}.")
        return SimulatedModel(name)
    kwargs: dict[str, Any] = {"timeout": timeout_s}
    if model:
        kwargs["model"] = model
    return get_provider(provider, **kwargs)


@dataclass
class Totals:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_eur: float = 0.0
    cost_known: bool = True
    ttft_ms: list[float] = field(default_factory=list)


@dataclass
class TurnResult:
    answer: str  # what the user sees (placeholders restored locally)
    completion: Completion  # what the model returned (placeholders inside)
    prompt: list[Message]  # exactly what was sent, after redaction and truncation
    redactions: dict[str, int]
    injection_flags: list[str]
    output_flags: list[str]
    attempts: list[Attempt]
    ttft_ms: float | None
    cost_eur: float | None
    trace: dict[str, Any]

    @property
    def decode_tps(self) -> float | None:
        if self.ttft_ms is None:
            return None
        return decode_tokens_per_second(
            self.completion.completion_tokens, self.ttft_ms, self.completion.latency_ms
        )


class Playground:
    """One chat session: settings, conversation state, gateway, tracer and running totals."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tracer: TraceWriter | None = None,
        providers: Sequence[LLMProvider] | None = None,
        clock: Clock | None = None,
        session_id: str = "cli",
    ) -> None:
        self.settings = settings or Settings()
        self.tracer = tracer or TraceWriter()
        self.session_id = session_id
        self.providers = list(providers) if providers else self._default_providers()
        simulated = all(isinstance(getattr(p, "inner", p), SimulatedModel) for p in self.providers)
        self.clock = clock or (VirtualClock() if simulated else RealClock())  # real back-ends really wait
        self._check_boundary()
        self.gateway = ModelGateway(
            self.providers, RetryPolicy(timeout_s=self.settings.timeout_s), self.clock
        )
        self.conversation = Conversation(system=self.settings.system)
        self.pseudo = Pseudonymiser(self.settings.redactor_level)
        self.totals = Totals()
        self.turn = 0

    def _default_providers(self) -> list[LLMProvider]:
        s = self.settings
        primary = make_provider(s.provider, s.model, s.timeout_s)
        if s.flaky:
            primary = FlakyProvider(primary, script=list(s.flaky))
        chain = [primary]
        if s.fallback_model:
            chain.append(make_provider(s.provider, s.fallback_model, s.timeout_s))
        return chain

    def _check_boundary(self) -> None:
        if not self.settings.redact and any(is_external(p) for p in self.providers):
            raise DataBoundaryError("Redaction is off but a provider is external. Turn redaction on.")

    @property
    def model(self) -> str:
        return self.providers[0].model

    @property
    def context_window(self) -> int:
        primary = getattr(self.providers[0], "inner", self.providers[0])
        profile = getattr(primary, "profile", None)
        return self.settings.context_window or (profile.context_window if profile else DEFAULT_CONTEXT_WINDOW)

    def prices_for(self, model: str) -> Prices | None:
        return self.settings.prices or EXAMPLE_PRICES.get(model)

    def reconfigure(self, **changes: Any) -> None:
        """Change settings mid-session (used by /model, /temp, /system). History is kept."""
        for key, value in changes.items():
            setattr(self.settings, key, value)
        if "system" in changes:
            self.conversation.system = self.settings.system
        if {"provider", "model", "fallback_model"} & changes.keys():
            self.providers = self._default_providers()
            self._check_boundary()
            self.gateway = ModelGateway(
                self.providers, RetryPolicy(timeout_s=self.settings.timeout_s), self.clock
            )

    def reset(self) -> None:
        self.conversation.reset()
        self.totals = Totals()

    # ------------------------------------------------------------------ #

    def ask(self, user_text: str) -> TurnResult:
        """Run one request through the whole lifecycle."""
        s = self.settings
        self.turn += 1
        # 1. input guardrails
        injection = scan_injection(user_text)
        if s.redact:
            safe_text, counts = self.pseudo.redact(user_text)
        else:
            safe_text, counts = user_text, {}
        # 2. prompt assembly
        budget = self.context_window - s.max_tokens
        messages = self.conversation.build(safe_text, policy=s.policy, budget=budget)
        base = {
            "trace_id": text_hash(f"{self.session_id}:{self.turn}:{safe_text}"),
            "session": self.session_id,
            "turn": self.turn,
            "params": {
                "temperature": s.temperature,
                "max_tokens": s.max_tokens,
                "seed": s.seed,
                "policy": s.policy,
            },
            "messages": len(messages),
            "prompt_tokens_est": prompt_tokens(messages),
            "redactions": dict(counts),
            "injection_flags": injection,
            "prompt_hash": text_hash("\n".join(f"{m.role}:{m.content}" for m in messages)),
        }
        # 3. model gateway
        try:
            result = self.gateway.complete(
                messages, temperature=s.temperature, max_tokens=s.max_tokens, seed=s.seed
            )
        except GatewayError as err:
            self.tracer.write(
                {**base, "error": err.kind.value, "attempts": [a.__dict__ for a in err.attempts]}
            )
            raise
        c = result.completion
        # 4. output checks
        output_flags = check_output(c, placeholders=self.pseudo.mapping)
        # 5. store, account and trace
        answer = self.pseudo.restore(c.text) if s.redact else c.text
        self.conversation.record(safe_text, c.text)
        prices = self.prices_for(c.model)
        cost = request_cost(c.prompt_tokens, c.completion_tokens, prices) if prices else None
        ttft = c.raw.get("ttft_ms") if isinstance(c.raw, dict) else None
        self._account(c, cost, ttft)
        trace = self.tracer.write(
            {
                **base,
                "provider": c.provider,
                "model": c.model,
                "external": is_external(self.providers[0]),
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "ttft_ms": ttft,
                "latency_ms": round(c.latency_ms, 3),
                "elapsed_s": round(result.elapsed_s, 3),
                "cost_eur": cost,
                "finish_reason": c.finish_reason,
                "attempts": [a.__dict__ for a in result.attempts],
                "fallback_used": result.fallback_used,
                "output_flags": output_flags,
                "error": None,
            }
        )
        return TurnResult(
            answer, c, messages, dict(counts), injection, output_flags, result.attempts, ttft, cost, trace
        )

    def _account(self, c: Completion, cost: float | None, ttft: float | None) -> None:
        t = self.totals
        t.requests += 1
        t.prompt_tokens += c.prompt_tokens
        t.completion_tokens += c.completion_tokens
        if cost is None:
            t.cost_known = False
        else:
            t.cost_eur += cost
        if ttft is not None:
            t.ttft_ms.append(ttft)
