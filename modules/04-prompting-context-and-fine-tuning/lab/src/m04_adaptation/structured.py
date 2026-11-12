"""Structured output: parse the model's answer, validate it against the label set, retry once."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .icl import RETRY_PREFIX
from .providers import LLMProvider, Message

_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


@dataclass(frozen=True)
class Parsed:
    """Result of parsing one answer: a valid label, or an error message to send back."""

    label: str | None
    error: str | None


def parse_answer(text: str, labels: Sequence[str], json_mode: bool = True) -> Parsed:
    """Accept only a known label. Tolerate prose around the JSON and a final 'Answer:' line."""
    if "Answer:" in text:
        text = text.rsplit("Answer:", 1)[1]
    text = text.strip()
    if json_mode:
        match = _JSON_OBJECT.search(text)
        if match is None:
            return Parsed(None, "no JSON object found")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return Parsed(None, f"invalid JSON ({exc.msg})")
        value = data.get("intent") if isinstance(data, dict) else None
        if not isinstance(value, str):
            return Parsed(None, 'missing string field "intent"')
        text = value
    if text in labels:
        return Parsed(text, None)
    return Parsed(None, f"{text!r} is not one of the allowed intents")


@dataclass(frozen=True)
class Classification:
    """Final label plus everything the evaluation report needs."""

    label: str | None
    attempts: int
    valid_first_try: bool
    prompt_tokens: int
    completion_tokens: int


def classify(
    provider: LLMProvider,
    messages: list[Message],
    labels: Sequence[str],
    *,
    json_mode: bool = True,
    max_retries: int = 1,
) -> Classification:
    """Call the model; if the answer is invalid, show it the error and ask again."""
    convo = list(messages)
    prompt_tokens = completion_tokens = 0
    first_ok = False
    for attempt in range(1, max_retries + 2):
        out = provider.complete(convo, temperature=0.0, max_tokens=120, seed=0)
        prompt_tokens += out.prompt_tokens
        completion_tokens += out.completion_tokens
        parsed = parse_answer(out.text, labels, json_mode)
        if attempt == 1:
            first_ok = parsed.label is not None
        if parsed.label is not None:
            return Classification(parsed.label, attempt, first_ok, prompt_tokens, completion_tokens)
        fmt = 'JSON only: {"intent": "<intent>"}' if json_mode else "the intent name only"
        convo += [
            Message("assistant", out.text),
            Message("user", f"{RETRY_PREFIX} was invalid: {parsed.error}. Reply with {fmt}, "
                            f"using one of: {', '.join(labels)}."),
        ]
    return Classification(None, max_retries + 1, first_ok, prompt_tokens, completion_tokens)
