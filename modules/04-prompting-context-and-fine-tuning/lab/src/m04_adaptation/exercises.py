"""Your turn: four small functions to implement while you follow the lessons.

Each one raises ``NotImplementedError`` until you write it. ``tests/test_exercises.py``
skips the matching test until then, and checks your version once it exists::

    uv run pytest tests/test_exercises.py -v

Reference solutions are in lab/README.md (hidden in <details>). Try first.
"""

from __future__ import annotations

import numpy as np

from .evaluation import ColumnAudit
from .prompts import ExampleSelector
from .tickets import Ticket


def flag_leaky_columns(audits: list[ColumnAudit], threshold: float = 0.95) -> list[str]:
    """Lesson 1. Return the names of suspicious columns, in input order.

    A column is suspicious when it is NOT known at prediction time, or when it predicts
    the label on its own with ``accuracy >= threshold``.
    """
    raise NotImplementedError


def select_diverse(selector: ExampleSelector, query: Ticket, k: int, diversity: float = 0.7) -> list[Ticket]:
    """Lesson 2. Pick k examples that are similar to the query but not to each other.

    Use maximal marginal relevance over the 4*k most similar candidates
    (``selector.select(query, 4 * k, "similar")``, which returns the most similar candidate LAST,
    so reverse it to break ties the same way). At each step choose the candidate c with
    the highest ``(1 - diversity) * sim(query, c) - diversity * max(sim(c, s) for s already chosen)``,
    where ``sim`` is the Jaccard overlap of ``stems(user_message(ticket, selector.cfg))``.
    With ``diversity = 0`` the result must contain the same tickets as plain similar selection.
    """
    raise NotImplementedError


def truncate_middle(text: str, max_tokens: int, marker: str = " [...] ") -> str:
    """Lesson 3. Shorten an over-long text by cutting words out of the middle.

    Keep the beginning and the end (where tickets usually state the problem and the request),
    join them with ``marker`` and return a text with ``count_tokens(result) <= max_tokens``.
    Return ``text`` unchanged when it already fits.
    """
    raise NotImplementedError


def lora_logits(
    X: np.ndarray, W: np.ndarray, b: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float
) -> np.ndarray:
    """Lesson 4. Logits of a LoRA model WITHOUT building the (C, D) update matrix.

    Compute ``X @ W.T + b`` plus the low-rank path ``(alpha / r) * (X @ A.T) @ B.T``, where
    r is the number of rows of A. The result must equal ``LoRAClassifier.logits(X)``.
    """
    raise NotImplementedError
