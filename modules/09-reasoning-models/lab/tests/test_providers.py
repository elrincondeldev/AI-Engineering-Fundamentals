"""Contract tests for the provider layer. They never touch the network."""

from __future__ import annotations

import math

import pytest

from m09_reasoning.providers import (
    Completion,
    FakeProvider,
    LLMProvider,
    Message,
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderError,
    ToolCall,
    count_tokens,
    get_provider,
    to_openai_messages,
)


def test_default_provider_is_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    provider = get_provider()
    assert provider.name == "fake"
    assert isinstance(provider, LLMProvider)


def test_fake_is_deterministic() -> None:
    msgs = [Message("system", "Be brief."), Message("user", "Hello there")]
    a = FakeProvider().complete(msgs)
    b = FakeProvider().complete(msgs)
    assert a.text == b.text
    assert a.text.startswith("[fake:")
    assert "Hello there" in a.text


def test_fake_responder_and_stop() -> None:
    provider = FakeProvider(responder=lambda msgs: "Answer: 42\nObservation: x")
    out = provider.complete([Message("user", "q")], stop=["\nObservation"])
    assert out.text == "Answer: 42"
    assert out.provider == "fake"


def test_fake_responder_can_return_tool_calls() -> None:
    call = ToolCall(id="c1", name="search", arguments={"q": "ember"})

    def responder(_msgs):
        return Completion("", "fake-1", "fake", 1, 0, 0.0, tool_calls=(call,), finish_reason="tool_calls")

    out = FakeProvider(responder=responder).complete([Message("user", "find it")])
    assert out.tool_calls == (call,)


def test_fake_embeddings_are_unit_length_and_lexical() -> None:
    a, b, c = FakeProvider().embed(["smart thermostat", "smart thermostat", "banana bread"])
    assert a == b
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
    same = sum(x * y for x, y in zip(a, b))
    diff = sum(x * y for x, y in zip(a, c))
    assert same > diff


def test_unknown_provider_raises() -> None:
    with pytest.raises(ProviderError):
        get_provider("nope")


def test_openai_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ProviderError):
        OpenAICompatibleProvider()


def test_payloads_are_built_offline() -> None:
    msgs = [
        Message("user", "weather?"),
        Message("assistant", "", tool_calls=(ToolCall("c1", "get_weather", {"city": "Valencia"}),)),
        Message("tool", "22C", name="get_weather", tool_call_id="c1"),
    ]
    oa = OpenAICompatibleProvider(model="m", base_url="http://localhost:1/v1").build_chat_payload(
        msgs, temperature=0.2, max_tokens=10, seed=7, stop=None, tools=None
    )
    assert oa["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Valencia"}'
    assert oa["seed"] == 7
    ol = OllamaProvider(model="m").build_chat_payload(
        msgs, temperature=0.2, max_tokens=10, seed=None, stop=["\n"], tools=None
    )
    assert ol["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"city": "Valencia"}
    assert ol["options"] == {"temperature": 0.2, "num_predict": 10, "stop": ["\n"]}


def test_serialisation_and_token_count() -> None:
    assert to_openai_messages([Message("user", "hi")]) == [{"role": "user", "content": "hi"}]
    assert count_tokens("Hello, world!") == 4
