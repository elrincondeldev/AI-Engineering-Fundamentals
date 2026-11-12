"""Prompt construction and example selection."""

from __future__ import annotations

from m04_adaptation.prompts import (
    ExampleSelector,
    PromptConfig,
    build_messages,
    descriptions,
    system_prompt,
    user_message,
)
from m04_adaptation.reports import standard_split
from m04_adaptation.textfeat import jaccard, stems
from m04_adaptation.tickets import INTENTS


def test_system_prompt_lists_every_intent_once() -> None:
    text = system_prompt(PromptConfig())
    for label in INTENTS:
        assert text.count(f"- {label}:") == 1
    assert "JSON" in text
    assert "JSON" not in system_prompt(PromptConfig(output="label"))
    assert all(v == "" for v in descriptions("names").values())
    assert "conectando" in descriptions("bilingual_v2")["connectivity_issue"]


def test_messages_have_one_pair_per_example() -> None:
    train, test = standard_split()
    cfg = PromptConfig(k=4)
    msgs = build_messages(test[0], cfg, ExampleSelector(train, cfg))
    assert [m.role for m in msgs] == ["system"] + ["user", "assistant"] * 4 + ["user"]


def test_selection_strategies() -> None:
    train, test = standard_split()
    cfg = PromptConfig(k=8)
    sel = ExampleSelector(train, cfg)
    query = test[0]
    similar = sel.select(query, 8, "similar")
    q = stems(user_message(query, cfg))
    sims = [jaccard(stems(user_message(t, cfg)), q) for t in similar]
    assert len(similar) == 8 and len(set(t.ticket_id for t in similar)) == 8
    assert sims[-1] >= sims[0]  # the closest example sits next to the query
    balanced = sel.select(query, 8, "balanced")
    assert sorted(t.intent for t in balanced) == sorted(INTENTS)
    assert sel.select(query, 8, "random") == sel.select(query, 8, "random")


def test_user_context_and_leak_flags() -> None:
    train, test = standard_split()
    t = test[0]
    with_ctx = build_messages(t, PromptConfig(user_context=True))[-1].content
    assert with_ctx.startswith("Customer context: language=") and "resolution_code" not in with_ctx
    leaky = build_messages(t, PromptConfig(leak_resolution_code=True))[-1].content
    assert f"resolution_code={t.resolution_code}" in leaky
