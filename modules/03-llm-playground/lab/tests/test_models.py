"""The simulated model: protocol, determinism, timing, context window and streaming."""

from __future__ import annotations

import random

import pytest

from m03_playground.errors import ErrorKind, PlaygroundError
from m03_playground.models import (
    PROFILES,
    SimulatedModel,
    candidates_for,
    chunks_from_completion,
    sample_index,
    simulate_timing,
)
from m03_playground.providers import FakeProvider, LLMProvider, Message

Q = [Message("system", "Be brief."), Message("user", "How long is the warranty?")]


def test_satisfies_the_provider_protocol() -> None:
    assert isinstance(SimulatedModel(), LLMProvider)
    assert SimulatedModel().name == "fake"


def test_same_input_same_output() -> None:
    a = SimulatedModel().complete(Q, temperature=1.2, seed=4)
    b = SimulatedModel().complete(Q, temperature=1.2, seed=4)
    assert a == b


def test_temperature_zero_is_greedy_whatever_the_seed() -> None:
    texts = {SimulatedModel().complete(Q, temperature=0.0, seed=s).text for s in range(20)}
    assert len(texts) == 1
    assert "2-year warranty" in texts.pop()


def test_high_temperature_explores() -> None:
    texts = {SimulatedModel().complete(Q, temperature=2.0, seed=s).text for s in range(30)}
    assert len(texts) >= 2


def test_sample_index_softmax() -> None:
    rng = random.Random(0)
    picks = [sample_index([2.0, 0.0], 1.0, rng) for _ in range(2000)]
    share = picks.count(0) / len(picks)
    assert 0.83 < share < 0.93  # exp(2) / (exp(2) + 1) = 0.881


def test_timing_formula() -> None:
    p = PROFILES["fake-small"]
    t = simulate_timing(p, prompt_tokens=100, completion_tokens=11)
    assert t.ttft_ms == pytest.approx(120 + 0.25 * 100)
    assert t.total_ms == pytest.approx(t.ttft_ms + 20 * 10)
    assert t.token_times(3) == pytest.approx([t.ttft_ms, (t.ttft_ms + t.total_ms) / 2, t.total_ms])


def test_prompt_tokens_match_fake_provider() -> None:
    sim = SimulatedModel().complete(Q)
    fake = FakeProvider().complete(Q)
    assert sim.prompt_tokens == fake.prompt_tokens


def test_context_window_is_enforced() -> None:
    with pytest.raises(PlaygroundError) as err:
        SimulatedModel("fake-small").complete(Q, max_tokens=1024)
    assert err.value.kind is ErrorKind.CONTEXT_LENGTH
    SimulatedModel("fake-large").complete(Q, max_tokens=1024)  # bigger window: fine


def test_max_tokens_truncates_and_reports_length() -> None:
    out = SimulatedModel().complete(Q, max_tokens=5)
    assert out.finish_reason == "length"


def test_stream_rebuilds_the_text_and_starts_at_ttft() -> None:
    model = SimulatedModel()
    done = model.complete(Q)
    chunks = list(model.stream(Q))
    assert "".join(c.text for c in chunks) == done.text
    assert len(chunks) == done.completion_tokens
    assert chunks[0].t_ms == done.raw["ttft_ms"]
    assert chunks[-1].t_ms == pytest.approx(done.latency_ms)
    assert list(chunks_from_completion(done)) == chunks


def test_unknown_topic_falls_back_to_generic_reply() -> None:
    cands = candidates_for("Write a haiku about Valencia")
    assert "Write a haiku about Valencia" in cands[0][0]


def test_one_sentence_system_prompt() -> None:
    msgs = [Message("system", "Answer in one sentence."), Message("user", "returns?")]
    text = SimulatedModel().complete(msgs).text
    assert text.count(". ") == 0 and text.endswith(".")
