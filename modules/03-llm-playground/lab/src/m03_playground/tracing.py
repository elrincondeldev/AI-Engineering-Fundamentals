"""JSONL traces: one JSON object per request, appended to a file.

JSON Lines is the simplest format that every tool can read (``jq``, pandas, a log
shipper). Rules for the playground, agreed with Noor and Sam:

- log *redacted* text only, or just a hash of it; never the raw prompt;
- log every number Sam needs to answer "why was that slow or expensive?";
- one line per request, written even when the request fails.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def text_hash(text: str) -> str:
    """Short, stable fingerprint: lets you group identical prompts without storing them."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass
class TraceWriter:
    """Append trace records to ``path`` (or keep them in memory when ``path`` is None)."""

    path: Path | None = None
    now: Callable[[], str] = utc_now
    records: list[dict[str, Any]] = field(default_factory=list)

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        line = {"ts": self.now(), **record}
        self.records.append(line)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False, sort_keys=False) + "\n")
        return line


def read_traces(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
