"""Locate the shared, read-only datasets in ``<repo>/datasets``.

Labs never import code from other modules, but they all read the same data.
Set ``COURSE_DATASETS_DIR`` to point somewhere else (for example in a fork).
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def datasets_dir() -> Path:
    """Return the datasets folder, searching upwards from this file."""
    override = os.environ.get("COURSE_DATASETS_DIR")
    if override:
        return Path(override).resolve()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "datasets"
        if (candidate / "generate.py").is_file():
            return candidate
    raise FileNotFoundError("Could not find the repo's datasets/ folder. Set COURSE_DATASETS_DIR.")


def read_jsonl(relative: str) -> list[dict[str, Any]]:
    """Read ``datasets/<relative>`` as a list of JSON objects."""
    path = datasets_dir() / relative
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_csv(relative: str) -> Iterator[dict[str, str]]:
    """Yield rows of ``datasets/<relative>`` as dictionaries."""
    path = datasets_dir() / relative
    with path.open(encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)
