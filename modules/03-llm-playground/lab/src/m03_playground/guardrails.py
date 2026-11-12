"""Input and output guardrails: data boundary, prompt-injection flags and output checks.

These are *tripwires*, not walls. Keyword rules catch known phrasings and miss
paraphrases; their job in the playground is to warn and to leave evidence in the
trace. The real protection is architectural: redact before data leaves, give the
model no secrets and no powers it does not need.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .data import datasets_dir
from .providers import Completion
from .redact import DIRECT, detect

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_external(provider: object) -> bool:
    """True when requests to this provider leave Hearthwise's machines. Unknown means external."""
    inner = getattr(provider, "inner", None)
    if inner is not None:  # wrappers such as FlakyProvider
        return is_external(inner)
    name = getattr(provider, "name", "")
    if name == "fake":
        return False
    url = getattr(provider, "host", None) or getattr(provider, "base_url", None)
    if name in ("ollama", "openai") and url:
        return (urlparse(url).hostname or "") not in LOCAL_HOSTS
    return True  # fail closed: if we cannot tell, treat it as external


# --------------------------------------------------------------------------- #
# Prompt-injection tripwires                                                  #
# --------------------------------------------------------------------------- #

INJECTION_RULES: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget)\b.{0,20}\b(previous|prior|above|earlier)\b.{0,15}\binstructions?\b",
        re.IGNORECASE,
    ),
    "system_prompt_probe": re.compile(
        r"\b(reveal|show|print|repeat)\b.{0,20}\b(system prompt|your instructions)\b", re.IGNORECASE
    ),
    "addressed_to_ai": re.compile(
        r"\b(note|message) to (ai|llm|the)? ?(assistants?|models?|ai)\b", re.IGNORECASE
    ),
    "hidden_html": re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE),
}


def scan_injection(text: str) -> list[str]:
    """Names of the injection rules that fire on ``text`` (empty list = nothing suspicious)."""
    return [name for name, rule in INJECTION_RULES.items() if rule.search(text)]


@dataclass(frozen=True)
class ScanResult:
    source: str
    flags: list[str]


def scan_knowledge_base() -> list[ScanResult]:
    """Run the tripwires over every help-centre document (Markdown and HTML)."""
    root = datasets_dir() / "kb"
    results = []
    for path in sorted(p for p in root.rglob("*") if p.suffix in {".md", ".html"}):
        text = path.read_text(encoding="utf-8")
        results.append(ScanResult(path.relative_to(root).as_posix(), scan_injection(text)))
    return results


def scan_texts(items: Iterable[tuple[str, str]]) -> list[ScanResult]:
    return [ScanResult(source, scan_injection(text)) for source, text in items]


# --------------------------------------------------------------------------- #
# Output checks                                                               #
# --------------------------------------------------------------------------- #

# Claims that are known to be wrong for Hearthwise (outdated 2023 returns policy, invented limits).
BANNED_CLAIMS: dict[str, re.Pattern[str]] = {
    "outdated_returns_policy": re.compile(r"within 14 days|10% restocking fee", re.IGNORECASE),
    "wrong_hub_capacity": re.compile(r"hub supports up to 250", re.IGNORECASE),
    "wrong_warranty": re.compile(r"warranty lasts 1 year", re.IGNORECASE),
}


def check_output(completion: Completion, *, placeholders: Iterable[str] = ()) -> list[str]:
    """Cheap checks on every answer before it is shown. Returns a list of flags."""
    flags: list[str] = []
    text = completion.text
    if not text.strip():
        flags.append("empty")
    if completion.finish_reason == "length":
        flags.append("truncated")
    known = set(placeholders)
    if any(s.kind in DIRECT and s.text not in known for s in detect(text, "names")):
        flags.append("pii_in_output")
    flags += [name for name, rule in BANNED_CLAIMS.items() if rule.search(text)]
    return flags


def kb_path(relative: str) -> Path:
    return datasets_dir() / "kb" / relative
