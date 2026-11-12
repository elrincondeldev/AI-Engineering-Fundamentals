#!/usr/bin/env python3
"""Check that the learner can walk the whole course with the footer links.

The expected chain, in English and in Spanish, is::

    docs/how-to-study → module README → lesson 1 → ... → last lesson → lab/README
    → quiz → next module README → ... → module 10 quiz → capstone

Usage::

    python3 tools/check_navigation.py            # both languages
    python3 tools/check_navigation.py --lang en
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")


def targets(path: Path, region: str) -> set[Path]:
    text = path.read_text(encoding="utf-8")
    if region == "footer":
        text = [line for line in text.splitlines() if line.strip()][-1]
    elif region == "next":
        start = text.find("⏭️")
        text = text[start:] if start >= 0 else ""
    return {(path.parent / m.group(1)).resolve() for m in LINK.finditer(text)}


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def check(lang: str) -> list[str]:
    sfx = ".es.md" if lang == "es" else ".md"
    modules = sorted(p for p in (REPO / "modules").iterdir() if p.is_dir() and p.name[:2].isdigit())
    problems: list[str] = []

    def page(folder: Path, stem: str) -> Path:
        return (folder / f"{stem}{sfx}").resolve()

    def expect(src: Path, region: str, want: Path, label: str) -> None:
        if not src.exists():
            problems.append(f"[{lang}] missing page {rel(src)}")
        elif want not in targets(src, region):
            problems.append(f"[{lang}] {rel(src)}: {label} link should point to {rel(want)}")

    for i, module in enumerate(modules):
        readme = page(module, "README")
        lessons = sorted(
            p.resolve() for p in module.glob(f"[0-9][0-9]-*{sfx}") if (lang == "es") == p.name.endswith(".es.md")
        )
        if not lessons:
            problems.append(f"[{lang}] {rel(module)}: no lessons")
            continue
        previous = page(modules[i - 1], "README") if i else page(REPO / "docs", "how-to-study")
        expect(readme, "footer", previous, "previous-module")
        expect(readme, "footer", lessons[0], "first-lesson")
        for j, lesson in enumerate(lessons):
            expect(lesson, "footer", readme if j == 0 else lessons[j - 1], "previous")
            expect(lesson, "footer", readme, "module-home")
            nxt = page(module / "lab", "README") if j == len(lessons) - 1 else lessons[j + 1]
            expect(lesson, "footer", nxt, "next")
        lab = page(module / "lab", "README")
        expect(lab, "footer", lessons[-1], "previous")
        expect(lab, "footer", page(module, "quiz"), "next")
        after = page(modules[i + 1], "README") if i + 1 < len(modules) else page(REPO / "capstone", "README")
        expect(page(module, "quiz"), "next", after, "⏭️ next")
    expect(page(REPO / "capstone", "README"), "footer", page(modules[-1], "quiz"), "previous")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["en", "es"], help="check one language only")
    args = ap.parse_args()
    problems = [p for lang in ([args.lang] if args.lang else ["en", "es"]) for p in check(lang)]
    for p in problems:
        print(f"❌ {p}")
    print(f"\n{len(problems)} navigation problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
