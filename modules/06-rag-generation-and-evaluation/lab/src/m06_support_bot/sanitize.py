"""Turn retrieved HTML and text into safe, plain data for the prompt.

Three defences live here (lesson 6.1):

1. ``html_to_text``      drops what a visitor never sees: hidden elements, scripts,
                         navigation and footers. A naive extractor keeps hidden text.
2. ``quarantine``        removes paragraphs that talk to the AI ("ignore all previous
                         instructions") even when they are visible.
3. ``escape_delimiters`` stops a document from closing our ``<document>`` tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier) (?:instructions|rules|messages)",
        r"\b(?:note|message|instructions?) (?:to|for) (?:the )?(?:ai|llm|chatbot|bot|assistant|model)s?\b",
        r"\b(?:ai|llm) assistants?\b.*\b(?:must|should|tell|say)\b",
        r"\bdisregard (?:all |any |your |the )?(?:previous |prior )?(?:instructions|rules)",
        r"\byou are now\b",
        r"\b(?:reveal|print|show) (?:your|the) (?:system )?prompt\b",
        r"\b(?:ignora|olvida) (?:todas )?las instrucciones",
    )
)

_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0", re.IGNORECASE)
_DELIMITER_RE = re.compile(r"<(/?)(documents?|question|instructions)\b", re.IGNORECASE)


@dataclass
class ExtractedHtml:
    """Text extracted from an HTML page, plus what was dropped because it was hidden."""

    text: str
    hidden: list[str] = field(default_factory=list)


class _Extractor(HTMLParser):
    """Collect visible blocks as Markdown-like lines: headings, paragraphs, table rows."""

    ALWAYS_SKIP = {"head", "script", "style", "nav", "footer", "noscript", "template"}
    VOID = {"br", "meta", "img", "input", "hr", "link"}
    BLOCKS = {"p", "li", "h1", "h2", "h3", "td", "th", "div", "tr"}

    def __init__(self, drop_hidden: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.drop_hidden = drop_hidden
        self.stack: list[tuple[str, bool, bool]] = []  # (tag, skipped, hidden)
        self.blocks: list[str] = []
        self.hidden: list[str] = []
        self.buf: list[str] = []
        self.hidden_buf: list[str] = []
        self.row: list[str] = []
        self.header: list[str] = []
        self.cell: list[str] | None = None

    # -- helpers -------------------------------------------------------------
    def _skipping(self) -> bool:
        return any(skip for _, skip, _ in self.stack)

    def _in_hidden(self) -> bool:
        return any(hidden for _, _, hidden in self.stack)

    def _flush(self, prefix: str = "") -> None:
        text = " ".join("".join(self.buf).split())
        if text:
            self.blocks.append(prefix + text)
        self.buf = []

    # -- HTMLParser hooks ----------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.VOID:
            return
        a = {k: (v or "") for k, v in attrs}
        hidden = bool(
            _HIDDEN_STYLE.search(a.get("style", ""))
            or "hidden" in a
            or a.get("aria-hidden", "").lower() == "true"
        )
        skip = tag in self.ALWAYS_SKIP or (hidden and self.drop_hidden)
        if tag in self.BLOCKS and not self._skipping():
            self._flush()
        if tag == "table":
            self.header = []
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = []
        self.stack.append((tag, skip, hidden))

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID or not self.stack:
            return
        # pop until the matching tag (tolerates sloppy HTML)
        while self.stack:
            open_tag, skip, hidden = self.stack.pop()
            if hidden and self.hidden_buf and not self._in_hidden():
                self.hidden.append(" ".join("".join(self.hidden_buf).split()))
                self.hidden_buf = []
            if open_tag == tag:
                break
        if self._skipping():
            return
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
            self.buf = []
        elif tag == "tr":
            self._end_row()
        elif tag in ("h1", "h2", "h3"):
            self._flush("#" * int(tag[1]) + " ")
        elif tag in self.BLOCKS:
            self._flush()

    def _end_row(self) -> None:
        cells = list(self.row)
        if cells and not self.header:
            self.header = cells
        elif cells:
            row = "; ".join(f"{h}: {v}" for h, v in zip(self.header, cells)).rstrip()
            self.blocks.append(row if row.endswith(".") else row + ".")
        self.row = []

    def handle_data(self, data: str) -> None:
        if self._in_hidden():
            self.hidden_buf.append(data)
        if self._skipping():
            return
        if self.cell is not None:
            self.cell.append(data)
        else:
            self.buf.append(data)

    def handle_comment(self, data: str) -> None:  # comments are never visible
        return


def html_to_text(html: str, *, drop_hidden: bool = True) -> ExtractedHtml:
    """Extract headings, paragraphs and linearised table rows from an HTML page.

    With ``drop_hidden=False`` the extractor behaves like a naive ``get_text()``:
    hidden paragraphs end up in the text and, later, in the prompt.
    """
    parser = _Extractor(drop_hidden=drop_hidden)
    parser.feed(html)
    parser.close()
    parser._flush()
    text = "\n\n".join(b for b in parser.blocks if b.strip())
    return ExtractedHtml(text=text, hidden=[h for h in parser.hidden if h])


def find_injections(text: str) -> list[str]:
    """Return the paragraphs that contain instruction-like text aimed at an AI."""
    found = []
    for paragraph in re.split(r"\n\s*\n", text):
        if any(p.search(paragraph) for p in INJECTION_PATTERNS):
            found.append(paragraph.strip())
    return found


def quarantine(text: str) -> tuple[str, list[str]]:
    """Remove injection paragraphs. Returns the clean text and what was removed."""
    removed = find_injections(text)
    if not removed:
        return text, []
    kept = [p for p in re.split(r"\n\s*\n", text) if p.strip() not in removed]
    return "\n\n".join(kept), removed


def escape_delimiters(text: str) -> str:
    """Neutralise our own prompt tags inside document text ('</document>' -> '&lt;/document>')."""
    return _DELIMITER_RE.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", text)
