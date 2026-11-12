"""Unit tests for the chat template, the loss mask and each data-quality check."""

from __future__ import annotations

import pytest

from m02_post_training.sft import (
    END,
    ContaminationIndex,
    chat_tokens,
    clean,
    detect_language,
    has_pii,
    is_empty,
    is_low_quality,
    is_outdated,
    is_wrong_language,
    load_sft,
    loss_mask,
    render_chat,
    score_against_labels,
)


def row(response: str, lang: str = "en", instruction: str = "q?", rid: str = "x1") -> dict[str, str]:
    return {"id": rid, "system": "", "instruction": instruction, "response": response, "lang": lang}


def test_render_chat_has_role_markers_and_generation_prompt() -> None:
    msgs = [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "Hi"}]
    text = render_chat(msgs, add_generation_prompt=True)
    assert text == "<|system|>\nBe kind.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n"
    with pytest.raises(ValueError):
        render_chat([{"role": "robot", "content": "x"}])


def test_loss_mask_trains_only_assistant_tokens_and_end_marker() -> None:
    msgs = [
        {"role": "user", "content": "Is it on?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Sure?"},
        {"role": "assistant", "content": "Yes!"},
    ]
    toks = chat_tokens(render_chat(msgs))
    trained = [t for t, m in zip(toks, loss_mask(toks)) if m]
    assert trained == ["Yes", ".", END, "Yes", "!", END]


def test_simple_checks() -> None:
    assert is_empty(row("   "))
    assert is_low_quality(row("asdf asdf asdf"))
    assert not is_low_quality(row("Yes, it needs a hub."))
    assert detect_language("Die Garantie beträgt zwei Jahre.") == "de"
    assert detect_language("Tienes 30 días desde la entrega.") == "es"
    assert is_wrong_language(row("Die Garantie beträgt zwei Jahre."))
    assert not is_wrong_language(row("Tienes 30 días desde la entrega.", lang="es"))
    assert has_pii(row("Write to ana@mail.example"))
    assert has_pii(row("Your card ending 4242 was charged."))
    assert not has_pii(row("We never need your full card number."))
    assert is_outdated(row("Opened items have a 10% restocking fee."))
    assert not is_outdated(row("You have 30 days from delivery."))


def test_contamination_index_needs_question_and_answer() -> None:
    idx = ContaminationIndex.from_rag_eval()
    copied = "What is the return window for Hearthwise products? 30 days from delivery."
    assert idx.matches(copied) == ["q001"]
    assert idx.matches("What is the return window for Hearthwise products? Ask support.") == []


def test_cleaner_finds_every_planted_problem() -> None:
    rows, labels = load_sft()
    report = clean(rows)
    for score in score_against_labels(report, labels):
        assert score.recall == 1.0, score.flag


def test_cleaner_drops_later_exact_duplicates() -> None:
    rows = [row("Yes.", rid="a"), row("Yes.", rid="b"), row("No.", rid="c")]
    report = clean(rows)
    assert [r["id"] for r in report.kept] == ["a", "c"]
    assert report.dropped == {"b": "duplicate"}
