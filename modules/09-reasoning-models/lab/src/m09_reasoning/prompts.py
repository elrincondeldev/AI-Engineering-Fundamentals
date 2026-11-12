"""System prompts and user-message formats.

The client (``reasoner.py``) and the simulator (``simulator.py``) share these strings, so the
simulator knows which task it is being asked to do. A real model receives exactly the same prompts.
"""

from __future__ import annotations

SOLVE_DIRECT = "Give only the final result. Reply with one line: 'Answer: <value>'."
SOLVE_COT = (
    "Solve the problem step by step. Write each step on its own line as 'Step k: ...'. "
    "Finish with one line: 'Answer: <value>'."
)
SOLVE_THINK = (
    "Think privately inside <think> ... </think> and check your work as you go. Then write the solution "
    "as 'Step k: ...' lines and finish with 'Answer: <value>'. Reasoning effort: {effort}."
)
NEXT_STEP = (
    "Continue the partial solution with exactly one line: the next 'Step k: ...', or "
    "'Answer: <value>' if the solution is already complete."
)
REVISE = (
    "Your previous solution was flagged. Read the feedback, then write a corrected solution as "
    "'Step k: ...' lines and finish with 'Answer: <value>'."
)
CRITIQUE = (
    "Check the solution below. If a step is wrong, reply 'Step k is wrong: <reason>'. "
    "Otherwise reply 'All steps are correct.'"
)
PROPOSE = (
    "You are solving a number puzzle. Propose different next moves. Each move combines two of the "
    "numbers left with +, -, * or /. Write one move per line as 'a op b = c'."
)
VALUE = (
    "Can the numbers left reach the target, using each exactly once with +, -, * and /? "
    "Reply with one word: sure, likely or impossible."
)
EFFORTS = ("low", "medium", "high")


def think_prompt(effort: str) -> str:
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}")
    return SOLVE_THINK.format(effort=effort)


def problem_message(question: str) -> str:
    return f"Problem: {question}"


def hint_message(question: str, answer: str) -> str:
    return f"Problem: {question}\nHint: the correct answer is {answer}."


def partial_message(question: str, steps: tuple[str, ...] | list[str]) -> str:
    body = "\n".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1)) or "(nothing yet)"
    return f"Problem: {question}\nSolution so far:\n{body}"


def solution_block(steps: tuple[str, ...] | list[str], answer: str | None) -> str:
    lines = [f"Step {i}: {s}" for i, s in enumerate(steps, 1)]
    lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def revise_message(question: str, steps: tuple[str, ...], answer: str | None, feedback: str) -> str:
    return f"Problem: {question}\nPrevious solution:\n{solution_block(steps, answer)}\nFeedback: {feedback}"


def critique_message(question: str, steps: tuple[str, ...], answer: str | None) -> str:
    return f"Problem: {question}\nSolution:\n{solution_block(steps, answer)}"


def numbers_message(question: str, left: str, k: int | None = None) -> str:
    wanted = f"\nMoves wanted: {k}" if k is not None else ""
    return f"Problem: {question}\nNumbers left: {left}{wanted}"
