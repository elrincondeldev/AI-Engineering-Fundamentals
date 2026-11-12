"""Live smoke tests against a local Ollama server. Skipped unless RUN_LIVE_TESTS=1.

Run them with:  RUN_LIVE_TESTS=1 OLLAMA_MODEL=llama3.2:1b uv run pytest -m ollama
They were NOT run when this lab was written (no model server was available).
"""

from __future__ import annotations

import pytest

from m10_deep_research.evaluate import load_workspace, run_and_grade
from m10_deep_research.loop import ResearchConfig
from m10_deep_research.planner import plan
from m10_deep_research.providers import Message, OllamaProvider


@pytest.mark.ollama
def test_ollama_answers_a_trivial_prompt() -> None:
    out = OllamaProvider().complete([Message("user", "Reply with the single word: ready")], max_tokens=5)
    assert out.text.strip() and out.provider == "ollama"


@pytest.mark.ollama
def test_ollama_planner_returns_valid_subqueries() -> None:
    subs = plan("Should Hearthwise launch the Ember T3 in Mexico, Colombia or Chile first?", OllamaProvider())
    assert 1 <= len(subs) <= 6 and all(s.query for s in subs)


@pytest.mark.ollama
def test_ollama_research_run_produces_a_gradable_report() -> None:
    ws = load_workspace()
    result, grade = run_and_grade(ws, "rq3", ResearchConfig(max_iterations=1), OllamaProvider())
    assert "## Sources" in result.report
    assert grade.facts_total == 2  # quality varies by model: inspect grade, do not assert a score
