"""Live round trips through the playground. Skipped unless RUN_LIVE_TESTS=1.

These tests were NOT run while the lab was written (no model server was available).
Run them yourself after following "Using a real model" in lab/README.md.
"""

from __future__ import annotations

import pytest

from m03_playground.playground import Playground, Settings
from m03_playground.tracing import TraceWriter

PROMPT = "Reply with one short sentence about smart thermostats."


@pytest.mark.ollama
def test_ollama_round_trip() -> None:
    pg = Playground(Settings(provider="ollama"), tracer=TraceWriter())
    result = pg.ask(PROMPT)
    assert result.completion.provider == "ollama"
    assert result.completion.text.strip()
    assert result.trace["external"] is False  # localhost is inside the data boundary


@pytest.mark.openai
def test_openai_compatible_round_trip() -> None:
    pg = Playground(Settings(provider="openai"), tracer=TraceWriter())
    result = pg.ask(PROMPT)
    assert result.completion.text.strip()
    assert result.completion.prompt_tokens > 0
