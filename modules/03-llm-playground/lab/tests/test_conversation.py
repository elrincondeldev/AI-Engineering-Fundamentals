"""Conversation state and truncation policies."""

from __future__ import annotations

from m03_playground.conversation import (
    KEY_FACT,
    POLICIES,
    SCRIPT,
    SUPPORT_SYSTEM,
    Conversation,
    growth,
    message_tokens,
    prompt_tokens,
    run_policy,
    summarise_turns,
)
from m03_playground.providers import FakeProvider, Message

SYSTEM = Message("system", SUPPORT_SYSTEM)


def _history(turns: int) -> list[Message]:
    out: list[Message] = []
    for user, assistant in SCRIPT[:turns]:
        out += [Message("user", user), Message("assistant", assistant)]
    return out


def test_token_count_matches_the_provider() -> None:
    msgs = [SYSTEM, *_history(3)]
    assert prompt_tokens(msgs) == FakeProvider().complete(msgs).prompt_tokens


def test_growth_is_cumulative() -> None:
    rows = growth()
    assert len(rows) == len(SCRIPT)
    assert all(b.prompt_tokens > a.prompt_tokens for a, b in zip(rows, rows[1:]))
    assert rows[-1].cumulative == sum(r.prompt_tokens for r in rows)


def test_policies_respect_the_budget_except_keep_all() -> None:
    history, user = _history(19), Message("user", SCRIPT[19][0])
    for name in ("pin_system", "summarise"):
        sent = POLICIES[name](SYSTEM, history, user, 768)
        assert prompt_tokens(sent) <= 768
        assert sent[0] == SYSTEM and sent[-1] == user
    assert prompt_tokens(POLICIES["keep_all"](SYSTEM, history, user, 768)) > 768


def test_sliding_window_loses_the_system_prompt() -> None:
    sent = POLICIES["sliding_window"](SYSTEM, _history(5), Message("user", "hi"), 768)
    assert len(sent) == 6
    assert SYSTEM not in sent


def test_pin_system_keeps_whole_pairs() -> None:
    sent = POLICIES["pin_system"](SYSTEM, _history(19), Message("user", "x"), 400)
    middle = sent[1:-1]
    assert middle and middle[0].role == "user" and middle[-1].role == "assistant"


def test_summary_keeps_hard_facts_within_its_cap() -> None:
    text = summarise_turns(_history(12), max_tokens=60)
    assert KEY_FACT in text and "Ember T3" in text
    assert message_tokens(Message("system", text)) <= 64


def test_policy_stats_shape() -> None:
    stats = {name: run_policy(name) for name in POLICIES}
    assert stats["keep_all"].over_budget_turns > 0
    assert stats["sliding_window"].turns_without_system > 0
    assert not stats["pin_system"].remembers_key_fact
    assert stats["summarise"].remembers_key_fact


def test_conversation_state() -> None:
    conv = Conversation(system="Be brief.")
    conv.record("hi", "hello")
    msgs = conv.build("again")
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
    conv.reset()
    assert conv.history == []
