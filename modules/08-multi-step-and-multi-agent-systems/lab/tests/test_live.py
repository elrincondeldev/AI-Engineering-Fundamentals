"""Live smoke tests against a real model. Skipped unless RUN_LIVE_TESTS=1.

They were not run while writing the lab (no model server was available). They check
only that the loop finishes and returns the expected format, never exact numbers.
"""

from __future__ import annotations

import pytest

from m08_web_agent.agents import run_react
from m08_web_agent.providers import OllamaProvider, OpenAICompatibleProvider
from m08_web_agent.search import Guards
from m08_web_agent.tools import WebTools

QUESTION = "How many devices can a single HomeMesh 2.0 network support?"


@pytest.mark.ollama
def test_react_with_ollama_finishes() -> None:
    r = run_react(QUESTION, WebTools(Guards()), OllamaProvider(), max_steps=6)
    assert r.answer_text.startswith("Final Answer:") and r.usage.calls >= 1


@pytest.mark.openai
def test_react_with_openai_compatible_finishes() -> None:
    r = run_react(QUESTION, WebTools(Guards()), OpenAICompatibleProvider(), max_steps=6)
    assert r.answer_text.startswith("Final Answer:") and r.usage.calls >= 1
