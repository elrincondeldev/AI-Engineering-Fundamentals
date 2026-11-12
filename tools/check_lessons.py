#!/usr/bin/env python3
"""Check that module pages follow the templates in docs/course-conventions.md.

It checks lessons (sections, word count, diagrams, callouts, self-check),
module READMEs, quizzes, cheatsheets and lab READMEs. Sections are found by
their emoji, so the same rules work for English and Spanish pages.

Usage::

    python3 tools/check_lessons.py                       # all modules, English
    python3 tools/check_lessons.py modules/05-rag-retrieval
    python3 tools/check_lessons.py --lang es modules/05-rag-retrieval
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _md import REPO, Parsed, parse, rel  # noqa: E402

LESSON_SECTIONS = ["📨", "🎯", "🤔", "#", "🧪", "🏭", "✅", "💬", "🔑", "📚"]
README_SECTIONS = ["🗺️", "📖", "🎯", "🗂️", "🧪", "📏", "🎤", "✅"]
LAB_SECTIONS = ["🎯", "⚙️", "👣", "💥", "🚀", "✅", "🧯", "🔌"]
MIN_WORDS, MAX_WORDS = 1800, 3500
# Spanish prose is naturally about 10-15% longer than the same content in English.
MAX_WORDS_ES = 4000
WORD_RE = re.compile(r"[\w'’-]+", re.UNICODE)


def h2(parsed: Parsed) -> list[tuple[int, str]]:
    return [(n, t) for n, level, t in parsed.headings if level == 2]


def section_text(parsed: Parsed, emoji: str) -> str:
    """Prose of the first level-2 section whose title starts with ``emoji``."""
    starts = h2(parsed)
    for idx, (line, title) in enumerate(starts):
        if title.startswith(emoji):
            end = starts[idx + 1][0] if idx + 1 < len(starts) else len(parsed.lines) + 1
            return "\n".join(t for n, t in parsed.prose_lines if line < n < end)
    return ""


def prose_words(parsed: Parsed) -> int:
    count = 0
    for _, line in parsed.prose_lines:
        s = line.strip()
        if not s or s.startswith("|") or s.startswith("#") or s.startswith("🌐") or s.startswith("[←"):
            continue
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"\]\([^)]*\)", "]", s)
        s = re.sub(r"`[^`]*`", " x ", s)
        count += len(WORD_RE.findall(s))
    return count


def check_order(parsed: Parsed, expected: list[str], what: str) -> list[str]:
    titles = [t for _, t in h2(parsed)]
    errors: list[str] = []
    pos = 0
    for marker in expected:
        if marker == "#":
            concepts = [t for t in titles if re.match(r"^\d+\.\s", t)]
            if len(concepts) < 2:
                errors.append(f"{what}: needs at least 2 numbered concept sections ('## 1. ...')")
            continue
        found = next((i for i in range(pos, len(titles)) if titles[i].startswith(marker)), None)
        if found is None:
            errors.append(f"{what}: missing or out-of-order section starting with {marker}")
        else:
            pos = found + 1
    return errors


def check_lesson(path: Path, lang: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    parsed = parse(text)
    errs = check_order(parsed, LESSON_SECTIONS, "structure")
    title = parsed.headings[0][2] if parsed.headings else ""
    word = "Lección" if lang == "es" else "Lesson"
    if not re.match(rf"^{word} \d+\.\d+ — ", title):
        errs.append(f"title must look like '# {word} 5.1 — ...'")
    words = prose_words(parsed)
    upper = MAX_WORDS_ES if lang == "es" else MAX_WORDS
    if not MIN_WORDS <= words <= upper:
        errs.append(f"{words} prose words; expected {MIN_WORDS}-{upper}")
    mermaid = sum(b.lang == "mermaid" for b in parsed.blocks)
    if mermaid < 2:
        errs.append(f"{mermaid} mermaid diagram(s); expected at least 2")
    if any(not b.lang for b in parsed.blocks):
        errs.append("every code block needs a language tag")
    if text.count("⚠️") < 2:
        errs.append("needs at least 2 '⚠️' common-mistake callouts")
    if text.count("💡") < 2:
        errs.append("needs at least 2 '💡' production-tip callouts")
    predict = section_text(parsed, "🤔")
    n_predict = len(re.findall(r"^\d+\.\s", predict, re.M))
    if not 2 <= n_predict <= 3:
        errs.append(f"'predict first' has {n_predict} numbered questions; expected 2-3")
    selfcheck = section_text(parsed, "✅")
    n_details = selfcheck.count("<details>")
    if not 4 <= n_details <= 6:
        errs.append(f"self-check has {n_details} <details> answers; expected 4-6")
    deeper = section_text(parsed, "📚")
    if "⭐" not in deeper:
        errs.append("'go deeper' must mark the best resource with ⭐")
    last = [line for line in parsed.lines if line.strip()][-1]
    if "🏠" not in last:
        errs.append("last line must be the navigation footer (with 🏠)")
    return errs


def check_readme(path: Path, module_no: str) -> list[str]:
    parsed = parse(path.read_text(encoding="utf-8"))
    errs = check_order(parsed, README_SECTIONS, "structure")
    maps = [b for b in parsed.blocks if b.lang == "mermaid" and "class M" in b.body]
    if not maps or f"class M{module_no} current" not in maps[0].body:
        errs.append(f"course map must highlight this module with 'class M{module_no} current'")
    interview = section_text(parsed, "🎤")
    if not 4 <= interview.count("<details>") <= 6:
        errs.append("interview corner needs 4-6 questions with <details> answers")
    rubric = section_text(parsed, "📏")
    if rubric.count("|") < 20:
        errs.append("self-assessment rubric should be a table with levels 1-4")
    if "[ ]" not in section_text(parsed, "🧪"):
        errs.append("lab section needs a Definition of Done checklist ('- [ ] ...')")
    return errs


def check_quiz(path: Path, module_no: str) -> list[str]:
    parsed = parse(path.read_text(encoding="utf-8"))
    qs = [t for _, level, t in parsed.headings if level == 3 and re.match(r"^[QP]\d+", t)]
    errs = []
    if not 10 <= len(qs) <= 15:
        errs.append(f"{len(qs)} questions; expected 10-15")
    review = sum("🔁" in t for t in qs)
    low = 0 if module_no == "01" else 2
    if not low <= review <= 3:
        errs.append(f"{review} spaced-repetition (🔁) questions; expected {low}-3")
    text = "\n".join(t for _, t in parsed.prose_lines)
    if text.count("<details>") < len(qs):
        errs.append("every question needs its answer in <details>")
    if "⏭️" not in text:
        errs.append("must end with a '## ⏭️' section linking to what comes next")
    return errs


def check_cheatsheet(path: Path) -> list[str]:
    n = len(path.read_text(encoding="utf-8").splitlines())
    return [f"{n} lines; a cheatsheet should fit one page (at most about 130 lines)"] if n > 130 else []


def check_lab(path: Path, lang: str) -> list[str]:
    parsed = parse(path.read_text(encoding="utf-8"))
    errs = check_order(parsed, LAB_SECTIONS, "structure")
    step = "Paso" if lang == "es" else "Step"
    steps = [t for _, level, t in parsed.headings if level == 3 and t.startswith(step)]
    if len(steps) < 3:
        errs.append(f"needs at least 3 '### {step} N — ...' headings")
    for emoji in ("💥", "🚀"):
        if "<details>" not in section_text(parsed, emoji):
            errs.append(f"section {emoji} should hide hints or solutions in <details>")
    if "[ ]" not in section_text(parsed, "✅"):
        errs.append("Definition of Done must be a checklist ('- [ ] ...')")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modules", nargs="*", help="module folders (default: all)")
    ap.add_argument("--lang", choices=["en", "es"], default="en")
    args = ap.parse_args()
    dirs = [REPO / m for m in args.modules] if args.modules else sorted((REPO / "modules").glob("[0-9][0-9]-*"))
    suffix = ".es.md" if args.lang == "es" else ".md"

    def pick(folder: Path, stem: str) -> Path:
        return folder / f"{stem}{suffix}"

    total = 0
    for d in dirs:
        no = d.name[:2]
        targets: list[tuple[Path, list[str]]] = []
        lessons = sorted(p for p in d.glob(f"[0-9][0-9]-*{suffix}") if (args.lang == "es") == p.name.endswith(".es.md"))
        if not lessons:
            targets.append((d, ["no lessons found"]))
        for lesson in lessons:
            targets.append((lesson, check_lesson(lesson, args.lang)))
        for stem, fn in (("README", lambda p: check_readme(p, no)), ("quiz", lambda p: check_quiz(p, no)),
                         ("cheatsheet", check_cheatsheet)):
            p = pick(d, stem)
            targets.append((p, fn(p) if p.exists() else ["file missing"]))
        lab = pick(d / "lab", "README")
        targets.append((lab, check_lab(lab, args.lang) if lab.exists() else ["file missing"]))
        for path, errs in targets:
            for e in errs:
                print(f"❌ {rel(path) if path.exists() else path}: {e}")
            total += len(errs)
        for lesson in lessons:
            print(f"   {rel(lesson)}: {prose_words(parse(lesson.read_text(encoding='utf-8')))} words")
    print(f"\n{total} problem(s) in {len(dirs)} module(s).")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
