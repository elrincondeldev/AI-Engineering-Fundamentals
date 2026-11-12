"""Your turn: three small functions to implement. ``tests/test_exercises.py`` checks them.

Each one raises ``NotImplementedError`` until you write it; the matching tests are skipped until then.
Solutions are in ``lab/README.md`` (hidden in <details>), but try first.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: the chance that at least one of k samples is correct, estimated from n samples
    of which c were correct (without replacement). Lesson 1.

    Examples: pass_at_k(10, 0, 3) == 0.0; pass_at_k(10, 10, 1) == 1.0; pass_at_k(5, 1, 1) == 0.2.
    Hint: 1 - (ways to pick k samples that are all wrong) / (ways to pick k samples).
    """
    raise NotImplementedError("implement pass_at_k in exercises.py")


def weighted_vote(answers: Sequence[str | None], scores: Sequence[float]) -> str | None:
    """Majority vote where each sample counts with its verifier score instead of 1. Lessons 2 and 3.

    Normalise answers with ``scaling.answer_key`` so '175.1' and '175.10' are the same answer.
    Ignore ``None`` answers. Ties go to the answer seen first. Return the answer as first written.
    """
    raise NotImplementedError("implement weighted_vote in exercises.py")


def adaptive_vote(draw: Callable[[], str | None], max_n: int, lead: int) -> tuple[str | None, int]:
    """Self-consistency that stops early: call ``draw()`` one sample at a time and stop as soon as the
    leading answer has ``lead`` more votes than the runner-up (or after ``max_n`` samples).
    Returns (majority answer, samples used). Lesson 2: the same accuracy for fewer tokens.
    """
    raise NotImplementedError("implement adaptive_vote in exercises.py")
