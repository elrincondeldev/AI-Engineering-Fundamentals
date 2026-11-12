"""Optional checks against a real model. Skipped unless RUN_LIVE_TESTS=1 (never run in CI)."""

from __future__ import annotations

import pytest

from m04_adaptation.prompts import ExampleSelector, PromptConfig, build_messages
from m04_adaptation.providers import OllamaProvider
from m04_adaptation.reports import standard_split
from m04_adaptation.structured import classify
from m04_adaptation.tickets import INTENTS


@pytest.mark.ollama
def test_ollama_returns_a_valid_intent() -> None:
    train, test = standard_split()
    cfg = PromptConfig(k=4)
    selector = ExampleSelector(train, cfg)
    provider = OllamaProvider()
    for ticket in test[:3]:
        result = classify(provider, build_messages(ticket, cfg, selector), INTENTS)
        assert result.label in INTENTS
