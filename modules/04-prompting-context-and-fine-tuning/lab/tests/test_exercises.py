"""Checks for the functions you write in src/m04_adaptation/exercises.py.

Each test is skipped while the exercise still raises NotImplementedError.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from m04_adaptation import exercises
from m04_adaptation.evaluation import ColumnAudit
from m04_adaptation.lora import LinearClassifier, LoRAClassifier
from m04_adaptation.prompts import ExampleSelector, PromptConfig, user_message
from m04_adaptation.providers import count_tokens
from m04_adaptation.reports import standard_split
from m04_adaptation.textfeat import jaccard, stems


def _try(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"exercise {fn.__name__} is not implemented yet")


def test_flag_leaky_columns() -> None:
    audits = [
        ColumnAudit("channel", 3, 0.12, True),
        ColumnAudit("magic_id", 900, 0.97, True),
        ColumnAudit("resolution_code", 8, 0.40, False),
    ]
    assert _try(exercises.flag_leaky_columns, audits) == ["magic_id", "resolution_code"]
    assert exercises.flag_leaky_columns(audits, threshold=0.99) == ["resolution_code"]


def test_select_diverse() -> None:
    train, test = standard_split()
    cfg = PromptConfig(k=4)
    sel = ExampleSelector(train, cfg)
    query = next(t for t in test if t.ticket_id == "T00822")
    plain = _try(exercises.select_diverse, sel, query, 4, diversity=0.0)
    assert {t.ticket_id for t in plain} == {t.ticket_id for t in sel.select(query, 4, "similar")}
    diverse = exercises.select_diverse(sel, query, 3, diversity=0.7)
    assert len({t.ticket_id for t in diverse}) == 3
    feats = [stems(user_message(t, cfg)) for t in diverse]
    assert all(jaccard(a, b) < 0.9 for i, a in enumerate(feats) for b in feats[i + 1:])


def test_truncate_middle() -> None:
    text = "My Lookout C2 keeps going offline. " + "blah " * 80 + "Please send a replacement."
    out = _try(exercises.truncate_middle, text, 30)
    assert count_tokens(out) <= 30
    assert out.startswith("My Lookout C2") and out.endswith("replacement.")
    assert "[...]" in out
    assert exercises.truncate_middle("short text", 30) == "short text"


def test_lora_logits() -> None:
    rng = np.random.default_rng(0)
    base = LinearClassifier(rng.normal(size=(8, 50)), rng.normal(size=8))
    m = LoRAClassifier(base, rng.normal(size=(2, 50)), rng.normal(size=(8, 2)), alpha=4.0)
    X = rng.random((5, 50))
    out = _try(exercises.lora_logits, X, base.W, base.b, m.A, m.B, m.alpha)
    assert np.allclose(out, m.logits(X))
