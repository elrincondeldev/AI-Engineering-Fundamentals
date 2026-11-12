"""Rule-based parsing of the Hearthwise help centre (Markdown with front matter, and HTML).

Both parsers turn a raw file into the same structure: a :class:`Document` with
metadata and a list of :class:`Block` objects (headings, text, tables). Keeping
the structure, instead of flattening everything to one string, is what makes
structure-aware and table-aware chunking possible later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from .data import datasets_dir, read_jsonl

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{3,}")


@dataclass(frozen=True)
class Block:
    """One structural element of a document."""

    kind: str  # "heading" | "text" | "table"
    text: str
    level: int = 0  # heading level (1 = title)
    rows: tuple[tuple[str, ...], ...] = ()  # table rows, first row is the header


@dataclass
class Document:
    """A parsed help-centre article."""

    doc_id: str
    title: str
    path: str
    format: str
    meta: dict[str, str]
    blocks: list[Block] = field(default_factory=list)
    hidden_dropped: int = 0  # HTML elements removed because they are invisible to readers

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


# --------------------------------------------------------------------------- #
# Markdown                                                                    #
# --------------------------------------------------------------------------- #


def split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split ``---``-delimited ``key: value`` front matter from the Markdown body."""
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.index("\n---\n", 4)
    meta: dict[str, str] = {}
    for line in raw[4:end].splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, raw[end + len("\n---\n") :]


def _table_row(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def table_to_text(rows: tuple[tuple[str, ...], ...]) -> str:
    """Flatten a table the naive way: every cell, row after row (headers are lost)."""
    return "\n".join(" ".join(r) for r in rows)


def parse_markdown_body(body: str) -> list[Block]:
    """Turn a Markdown body into heading, text and table blocks."""
    blocks: list[Block] = []
    para: list[str] = []
    table: list[tuple[str, ...]] = []

    def flush() -> None:
        if para:
            blocks.append(Block("text", " ".join(s.strip() for s in para)))
            para.clear()
        if table:
            rows = tuple(table)
            blocks.append(Block("table", table_to_text(rows), rows=rows))
            table.clear()

    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            if para:
                flush()
            if not TABLE_SEP_RE.match(stripped.strip("|").strip() or "x"):
                table.append(_table_row(stripped))
            continue
        if table:
            flush()
        heading = HEADING_RE.match(stripped)
        if heading:
            flush()
            blocks.append(Block("heading", heading.group(2), level=len(heading.group(1))))
        elif not stripped:
            flush()
        elif stripped.startswith(("- ", "* ")) or re.match(r"^\d+\.\s", stripped):
            flush()  # every list item is its own text block
            para.append(stripped)
        else:
            para.append(stripped)
    flush()
    return blocks


# --------------------------------------------------------------------------- #
# HTML                                                                        #
# --------------------------------------------------------------------------- #


class _HelpCentreHTML(HTMLParser):
    """Collect metadata and visible blocks; skip navigation, footers and hidden elements."""

    SKIP_TAGS = {"nav", "footer", "aside", "script", "style", "header", "head"}
    VOID = {"br", "meta", "img", "input", "hr", "link"}
    BLOCK_TAGS = {"p", "li", "div"}

    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self.title = ""
        self.blocks: list[Block] = []
        self.hidden_dropped = 0
        self._stack: list[tuple[str, bool]] = []
        self._skip_depth = 0
        self._buf: list[str] = []
        self._in_title = False
        self._rows: list[tuple[str, ...]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if tag == "meta" and "name" in a:
            self.meta[a["name"]] = a.get("content", "")
        if tag == "title":
            self._in_title = True
        if tag in self.VOID:
            return
        hidden = "display:none" in a.get("style", "").replace(" ", "") or "hidden" in a
        skip = tag in self.SKIP_TAGS or hidden or "cookie" in a.get("class", "")
        if hidden and self._skip_depth == 0:
            self.hidden_dropped += 1
        self._stack.append((tag, skip))
        if skip:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "table":
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag in self.BLOCK_TAGS or re.fullmatch(r"h[1-6]", tag):
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self.VOID or not self._stack:
            return
        open_tag, skip = self._stack.pop()
        if skip:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        text = " ".join(" ".join(self._buf).split())
        if open_tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join(" ".join(self._cell).split()))
            self._cell = None
        elif open_tag == "tr" and self._rows is not None and self._row is not None:
            self._rows.append(tuple(self._row))
            self._row = None
        elif open_tag == "table" and self._rows is not None:
            rows = tuple(self._rows)
            self.blocks.append(Block("table", table_to_text(rows), rows=rows))
            self._rows = None
        elif re.fullmatch(r"h[1-6]", open_tag) and text:
            self.blocks.append(Block("heading", text, level=int(open_tag[1])))
            self._buf = []
        elif open_tag in self.BLOCK_TAGS and text:
            self.blocks.append(Block("text", text))
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip()
            return
        if self._skip_depth or not data.strip():
            return
        if self._cell is not None:
            self._cell.append(data.strip())
        else:
            self._buf.append(data.strip())


def parse_html(raw: str) -> tuple[dict[str, str], str, list[Block], int]:
    """Return (meta, title, blocks, hidden elements dropped) for one HTML page."""
    parser = _HelpCentreHTML()
    parser.feed(raw)
    parser.close()
    return parser.meta, parser.title, parser.blocks, parser.hidden_dropped


# --------------------------------------------------------------------------- #
# Loading the knowledge base                                                  #
# --------------------------------------------------------------------------- #


def parse_file(meta: dict[str, str], raw: str) -> Document:
    """Parse one knowledge-base file described by its manifest entry."""
    if meta["format"] == "markdown":
        front, body = split_front_matter(raw)
        blocks = parse_markdown_body(body)
        merged = {**{k: str(v) for k, v in meta.items()}, **front}
        title = merged.get("title", meta["title"])
        return Document(meta["id"], title, meta["path"], "markdown", merged, blocks)
    page_meta, title, blocks, hidden = parse_html(raw)
    merged = {**{k: str(v) for k, v in meta.items()}, **page_meta}
    return Document(meta["id"], title or meta["title"], meta["path"], "html", merged, blocks, hidden)


def load_kb() -> list[Document]:
    """Parse every document listed in ``datasets/kb/manifest.jsonl``, in manifest order."""
    root = datasets_dir()
    docs = []
    for meta in read_jsonl("kb/manifest.jsonl"):
        raw = (root / meta["path"]).read_text(encoding="utf-8")
        docs.append(parse_file(meta, raw))
    return docs
