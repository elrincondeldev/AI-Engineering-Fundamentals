#!/usr/bin/env python3
"""Check that English and Spanish pages stay in sync.

For every course Markdown page ``X.md`` there must be an ``X.es.md`` (and vice
versa) with the same number of headings, fenced code blocks and mermaid
blocks, and both must start with the language selector.

Usage::

    python3 tools/check_i18n.py                 # whole repository
    python3 tools/check_i18n.py modules/05-rag-retrieval docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _md import is_spanish, markdown_files, parse, rel, sibling  # noqa: E402


def selector_ok(path: Path, first_line: str) -> str | None:
    en = path.name if not is_spanish(path) else sibling(path).name
    es = sibling(path).name if not is_spanish(path) else path.name
    expected = f"🌐 [English]({en}) · Español" if is_spanish(path) else f"🌐 English · [Español]({es})"
    if first_line.strip() != expected:
        return f"first line must be the language selector: {expected!r}"
    return None


def stats(path: Path) -> dict[str, int]:
    parsed = parse(path.read_text(encoding="utf-8"))
    return {
        "headings": len(parsed.headings),
        "code blocks": len(parsed.blocks),
        "mermaid blocks": sum(1 for b in parsed.blocks if b.lang == "mermaid"),
        "<details> blocks": sum(line.count("<details>") for _, line in parsed.prose_lines),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="files or folders (default: whole repository)")
    args = ap.parse_args()

    errors: list[str] = []
    files = markdown_files(args.paths or None)
    english = [p for p in files if not is_spanish(p)]
    spanish = [p for p in files if is_spanish(p)]

    for es in spanish:
        if not sibling(es).exists():
            errors.append(f"{rel(es)}: orphan translation, {sibling(es).name} does not exist")

    for en in english:
        es = sibling(en)
        first = en.read_text(encoding="utf-8").splitlines()[:1]
        msg = selector_ok(en, first[0] if first else "")
        if msg:
            errors.append(f"{rel(en)}: {msg}")
        if not es.exists():
            errors.append(f"{rel(en)}: missing Spanish version {es.name}")
            continue
        first = es.read_text(encoding="utf-8").splitlines()[:1]
        msg = selector_ok(es, first[0] if first else "")
        if msg:
            errors.append(f"{rel(es)}: {msg}")
        a, b = stats(en), stats(es)
        for key in a:
            if a[key] != b[key]:
                errors.append(f"{rel(es)}: {key} differ (en={a[key]}, es={b[key]})")

    for e in errors:
        print(f"❌ {e}")
    print(f"\n{len(english)} English pages, {len(spanish)} Spanish pages, {len(errors)} problem(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
