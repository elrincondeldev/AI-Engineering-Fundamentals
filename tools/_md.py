"""Small Markdown helpers shared by the checker scripts (standard library only)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXCLUDED_DIRS = {".git", ".venv", "node_modules", "outputs", "__pycache__", ".pytest_cache", ".ruff_cache"}
# Markdown files that are data, not course pages (the fictional help centre).
DATA_PREFIXES = ("datasets/kb/",)

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


@dataclass
class Block:
    """A fenced code block."""

    lang: str
    start: int  # 1-based line number of the opening fence
    body: str


@dataclass
class Parsed:
    lines: list[str]
    prose_lines: list[tuple[int, str]]  # (line number, text) outside fenced code
    blocks: list[Block]
    headings: list[tuple[int, int, str]]  # (line number, level, text)


def parse(text: str) -> Parsed:
    lines = text.splitlines()
    prose: list[tuple[int, str]] = []
    blocks: list[Block] = []
    headings: list[tuple[int, int, str]] = []
    fence: str | None = None
    lang = ""
    start = 0
    body: list[str] = []
    for i, line in enumerate(lines, 1):
        m = FENCE_RE.match(line)
        if fence is None:
            if m:
                fence = m.group(2)
                lang = m.group(3).strip().split(" ")[0].lower()
                start = i
                body = []
                continue
            prose.append((i, line))
            h = HEADING_RE.match(line)
            if h:
                headings.append((i, len(h.group(1)), h.group(2)))
        else:
            if m and m.group(2)[0] == fence[0] and len(m.group(2)) >= len(fence) and not m.group(3).strip():
                blocks.append(Block(lang=lang, start=start, body="\n".join(body)))
                fence = None
                continue
            body.append(line)
    if fence is not None:
        blocks.append(Block(lang=lang, start=start, body="\n".join(body)))
    return Parsed(lines=lines, prose_lines=prose, blocks=blocks, headings=headings)


def github_slug(heading: str) -> str:
    """Approximate GitHub's heading anchor algorithm."""
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("`", "").strip().lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "-_ " or cat[0] in ("L", "N") or cat == "Mn":
            out.append(ch)
    return "".join(out).replace(" ", "-")


def anchors(parsed: Parsed) -> set[str]:
    seen: dict[str, int] = {}
    result: set[str] = set()
    for _, _, text in parsed.headings:
        slug = github_slug(text)
        if slug in seen:
            seen[slug] += 1
            result.add(f"{slug}-{seen[slug]}")
        else:
            seen[slug] = 0
            result.add(slug)
    for _, line in parsed.prose_lines:
        for m in re.finditer(r'<a\s+(?:id|name)="([^"]+)"', line):
            result.add(m.group(1))
    return result


def markdown_files(paths: list[str] | None = None) -> list[Path]:
    roots = [REPO / p for p in paths] if paths else [REPO]
    found: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".md":
            found.add(root.resolve())
            continue
        for path in root.rglob("*.md"):
            rel = path.resolve().relative_to(REPO).as_posix()
            if any(part in EXCLUDED_DIRS for part in path.parts) or rel.startswith(DATA_PREFIXES):
                continue
            found.add(path.resolve())
    return sorted(found)


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def is_spanish(path: Path) -> bool:
    return path.name.endswith(".es.md")


def sibling(path: Path) -> Path:
    if is_spanish(path):
        return path.with_name(path.name[: -len(".es.md")] + ".md")
    return path.with_name(path.name[: -len(".md")] + ".es.md")
