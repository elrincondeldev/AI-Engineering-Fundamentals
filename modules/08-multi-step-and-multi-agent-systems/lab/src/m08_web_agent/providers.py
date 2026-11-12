"""Provider-agnostic LLM interface shared (by copy) across every lab.

Three back-ends sit behind one small Protocol:

- ``fake``   deterministic and offline. The default. Tests and CI use it.
- ``ollama`` a local model served by Ollama (http://localhost:11434).
- ``openai`` any OpenAI-compatible endpoint (OpenAI, vLLM, LM Studio,
             llama.cpp server, ...).

Pick one with the ``LLM_PROVIDER`` environment variable::

    LLM_PROVIDER=fake    # default, no network
    LLM_PROVIDER=ollama  OLLAMA_MODEL=llama3.2:1b
    LLM_PROVIDER=openai  OPENAI_BASE_URL=... OPENAI_API_KEY=... OPENAI_MODEL=...

Only the Python standard library is used, so this file can be copied into any
lab without adding dependencies. Labs customise the fake provider by passing a
``responder`` callable instead of editing this file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class ProviderError(RuntimeError):
    """Raised when a provider cannot complete a request (network, config, API error)."""


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to run one tool with JSON arguments."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """One chat message. ``tool_calls`` is only used on assistant messages."""

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class Completion:
    """What every provider returns, whatever the back-end."""

    text: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@runtime_checkable
class LLMProvider(Protocol):
    """The contract all labs program against."""

    name: str
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Completion: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def count_tokens(text: str) -> int:
    """Approximate token count: words and punctuation marks.

    Real tokenizers (BPE, SentencePiece) give different numbers; use this only
    for budgeting and tests, never for billing.
    """
    return len(_TOKEN_RE.findall(text))


def _messages_tokens(messages: Sequence[Message]) -> int:
    return sum(count_tokens(m.content) + 4 for m in messages)


def _stable_digest(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Fake provider                                                               #
# --------------------------------------------------------------------------- #

Responder = Callable[[Sequence[Message]], "str | Completion"]


def hashing_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic bag-of-words + character-trigram embedding (feature hashing).

    It captures lexical overlap only: synonyms and other languages do not match.
    That limitation is deliberate and is discussed in the RAG modules.
    """
    vec = [0.0] * dim
    words = [w.lower() for w in re.findall(r"\w+", text, re.UNICODE)]
    features = list(words)
    for w in words:
        padded = f"#{w}#"
        features.extend(padded[i : i + 3] for i in range(len(padded) - 2))
    for feat in features:
        d = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(d[:4], "little") % dim
        sign = 1.0 if d[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class FakeProvider:
    """Deterministic offline provider: the same input always gives the same output.

    Without a ``responder`` it returns a short canned reply that mentions the
    last user message. Labs pass a ``responder`` to script behaviour, for
    example an extractive answerer or a policy that emits tool calls. A
    responder may return a plain string or a full :class:`Completion`.
    """

    name = "fake"

    def __init__(
        self,
        responder: Responder | None = None,
        *,
        model: str = "fake-1",
        embedding_dim: int = 256,
    ) -> None:
        self.model = model
        self.embedding_dim = embedding_dim
        self._responder = responder
        self.calls: list[list[Message]] = []

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
        if not messages:
            raise ProviderError("messages must not be empty")
        self.calls.append(list(messages))
        start = time.perf_counter()
        if self._responder is not None:
            out = self._responder(messages)
            if isinstance(out, Completion):
                return out
            text = out
        else:
            last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            digest = _stable_digest(*(f"{m.role}:{m.content}" for m in messages))[:8]
            preview = " ".join(last_user.split())[:60]
            text = f"[fake:{digest}] You said: {preview}"
        if stop:
            for s in stop:
                cut = text.find(s)
                if cut != -1:
                    text = text[:cut]
        words = text.split(" ")
        finish = "stop"
        if len(words) > max_tokens:
            text, finish = " ".join(words[:max_tokens]), "length"
        return Completion(
            text=text,
            model=self.model,
            provider=self.name,
            prompt_tokens=_messages_tokens(messages),
            completion_tokens=count_tokens(text),
            latency_ms=(time.perf_counter() - start) * 1000,
            finish_reason=finish,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [hashing_embedding(t, self.embedding_dim) for t in texts]


# --------------------------------------------------------------------------- #
# HTTP helpers                                                                #
# --------------------------------------------------------------------------- #


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (URL comes from config)
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(f"HTTP {exc.code} from {url}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ProviderError(f"Cannot reach {url}: {exc}") from exc


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Model returned invalid JSON tool arguments: {raw[:200]}") from exc
        if isinstance(parsed, dict):
            return parsed
    return {}


def to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Serialise messages to the OpenAI chat format (also accepted by Ollama)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        item: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.name and m.role == "tool":
            item["name"] = m.name
        if m.tool_call_id:
            item["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            item["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Ollama                                                                      #
# --------------------------------------------------------------------------- #


class OllamaProvider:
    """Local models through Ollama's HTTP API (``/api/chat`` and ``/api/embed``)."""

    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        *,
        host: str | None = None,
        embed_model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
        self.embed_model = embed_model or os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def build_chat_payload(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        stop: Sequence[str] | None,
        tools: Sequence[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed
        if stop:
            options["stop"] = list(stop)
        msgs = to_openai_messages(messages)
        for msg in msgs:  # Ollama expects arguments as objects, not JSON strings
            for tc in msg.get("tool_calls", []):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
        payload: dict[str, Any] = {"model": self.model, "messages": msgs, "stream": False, "options": options}
        if tools:
            payload["tools"] = list(tools)
        return payload

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
        payload = self.build_chat_payload(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed, stop=stop, tools=tools
        )
        start = time.perf_counter()
        data = _post_json(f"{self.host}/api/chat", payload, {}, self.timeout)
        latency = (time.perf_counter() - start) * 1000
        msg = data.get("message", {})
        calls = tuple(
            ToolCall(
                id=f"call_{i}",
                name=tc.get("function", {}).get("name", ""),
                arguments=_parse_arguments(tc.get("function", {}).get("arguments")),
            )
            for i, tc in enumerate(msg.get("tool_calls") or [])
        )
        return Completion(
            text=msg.get("content", ""),
            model=data.get("model", self.model),
            provider=self.name,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            latency_ms=latency,
            tool_calls=calls,
            finish_reason=data.get("done_reason", "stop"),
            raw=data,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        data = _post_json(
            f"{self.host}/api/embed", {"model": self.embed_model, "input": list(texts)}, {}, self.timeout
        )
        return data["embeddings"]


# --------------------------------------------------------------------------- #
# OpenAI-compatible                                                           #
# --------------------------------------------------------------------------- #


class OpenAICompatibleProvider:
    """Any server that speaks the OpenAI ``/chat/completions`` and ``/embeddings`` API."""

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        embed_model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        resolved = model or os.environ.get("OPENAI_MODEL")
        if not resolved:
            raise ProviderError("Set OPENAI_MODEL (for example the model name your endpoint serves).")
        self.model = resolved
        self.embed_model = embed_model or os.environ.get("OPENAI_EMBED_MODEL", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def build_chat_payload(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        stop: Sequence[str] | None,
        tools: Sequence[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if stop:
            payload["stop"] = list(stop)
        if tools:
            payload["tools"] = list(tools)
        return payload

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
        payload = self.build_chat_payload(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed, stop=stop, tools=tools
        )
        start = time.perf_counter()
        data = _post_json(f"{self.base_url}/chat/completions", payload, self._headers(), self.timeout)
        latency = (time.perf_counter() - start) * 1000
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected response shape: {str(data)[:300]}") from exc
        msg = choice.get("message", {})
        calls = tuple(
            ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=tc.get("function", {}).get("name", ""),
                arguments=_parse_arguments(tc.get("function", {}).get("arguments")),
            )
            for i, tc in enumerate(msg.get("tool_calls") or [])
        )
        usage = data.get("usage", {})
        return Completion(
            text=msg.get("content") or "",
            model=data.get("model", self.model),
            provider=self.name,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency,
            tool_calls=calls,
            finish_reason=choice.get("finish_reason") or "stop",
            raw=data,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.embed_model:
            raise ProviderError("Set OPENAI_EMBED_MODEL to use embeddings with this endpoint.")
        data = _post_json(
            f"{self.base_url}/embeddings",
            {"model": self.embed_model, "input": list(texts)},
            self._headers(),
            self.timeout,
        )
        return [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def get_provider(name: str | None = None, **kwargs: Any) -> LLMProvider:
    """Return the provider named by ``name`` or ``$LLM_PROVIDER`` (default: ``fake``)."""
    chosen = (name or os.environ.get("LLM_PROVIDER") or "fake").strip().lower()
    if chosen == "fake":
        return FakeProvider(**kwargs)
    if chosen == "ollama":
        return OllamaProvider(**kwargs)
    if chosen == "openai":
        return OpenAICompatibleProvider(**kwargs)
    raise ProviderError(f"Unknown provider {chosen!r}. Use one of: fake, ollama, openai.")
