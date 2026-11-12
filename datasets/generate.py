#!/usr/bin/env python3
"""Generate every synthetic dataset used in the course.

Usage::

    python datasets/generate.py           # (re)write all files under datasets/
    python datasets/generate.py --check   # verify the committed files are up to date

Standard library only. The output is byte-for-byte deterministic: the same
seed always produces the same files, on any machine and any Python >= 3.12.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from _gen.kb import build_kb, kb_manifest, rag_eval  # noqa: E402
from _gen.prefs_sft import build_preferences, build_sft  # noqa: E402
from _gen.reasoning import build_reasoning  # noqa: E402
from _gen.tickets import build_tickets  # noqa: E402
from _gen.web import agent_tasks, build_web, research_questions  # noqa: E402


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def build_all() -> dict[str, str]:
    """Return {path relative to datasets/: file content}."""
    files: dict[str, str] = {}
    kb = build_kb()
    for rel, content in kb.items():
        files[f"kb/{rel}"] = content
    files["kb/manifest.jsonl"] = _jsonl(kb_manifest(kb))

    qa = rag_eval()
    files["rag_eval/qa.jsonl"] = _jsonl(qa)

    reasoning = build_reasoning()
    files["reasoning/problems.jsonl"] = _jsonl(reasoning)

    pages, labels = build_web(qa, reasoning)
    files["web/pages.jsonl"] = _jsonl(pages)
    files["web/labels.jsonl"] = _jsonl(labels)

    files["agent_tasks/tasks.jsonl"] = _jsonl(agent_tasks())
    files["agent_tasks/research_questions.jsonl"] = _jsonl(research_questions())

    files["tickets/tickets.csv"] = _csv(build_tickets())
    files["preferences/pairs.jsonl"] = _jsonl(build_preferences())
    sft, sft_labels = build_sft(qa)
    files["sft/instructions.jsonl"] = _jsonl(sft)
    files["sft/labels.jsonl"] = _jsonl(sft_labels)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if committed files differ from a fresh build")
    args = parser.parse_args()

    files = build_all()
    if args.check:
        stale = []
        for rel, content in sorted(files.items()):
            path = ROOT / rel
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(rel)
        if stale:
            print("Datasets are out of date. Run: python datasets/generate.py")
            for rel in stale:
                print(f"  - {rel}")
            return 1
        print(f"OK: {len(files)} dataset files are up to date.")
        return 0

    for rel, content in sorted(files.items()):
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    print(f"Wrote {len(files)} files under {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
