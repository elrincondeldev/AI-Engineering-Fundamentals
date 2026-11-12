"""Text helpers: a tiny word-level tokenizer, HTML-to-text and the Hearthwise training corpus.

Real LLMs use subword tokenizers (module 01). Here a word-level tokenizer keeps every
example readable: one token is one word or one punctuation mark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from .data import datasets_dir, read_jsonl

TOKEN_RE = re.compile(r"\w+(?:[-.'’/]\w+)*|[^\w\s]", re.UNICODE)
_NO_SPACE_BEFORE = set(".,;:!?)%'’")
_NO_SPACE_AFTER = set("(€")


def tokenize(text: str) -> list[str]:
    """Lower-case word-level tokens: ``"Wi-Fi 2.4 GHz."`` -> ``["wi-fi", "2.4", "ghz", "."]``."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def detokenize(tokens: list[str]) -> str:
    """Join tokens back into readable text (no space before punctuation)."""
    out = ""
    for tok in tokens:
        if not out or tok in _NO_SPACE_BEFORE or out[-1] in _NO_SPACE_AFTER:
            out += tok
        else:
            out += " " + tok
    return out


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """All contiguous n-grams of a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class _TextExtractor(HTMLParser):
    """Keep visible article text; drop navigation, footers, cookie banners and hidden elements."""

    SKIP = {"nav", "footer", "aside", "script", "style", "header", "head"}
    VOID = {"br", "meta", "img", "input", "hr", "link"}

    def __init__(self) -> None:
        super().__init__()
        self.skip_depth = 0
        self.stack: list[bool] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.VOID:
            return
        a = dict(attrs)
        cls, style = a.get("class") or "", a.get("style") or ""
        skip = tag in self.SKIP or "cookie" in cls or cls == "ads" or "display:none" in style
        self.stack.append(skip)
        self.skip_depth += skip

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID or not self.stack:
            return
        self.skip_depth -= self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0 and data.strip():
            self.parts.append(data.strip())


def html_text(html: str) -> str:
    """Visible text of an HTML page."""
    parser = _TextExtractor()
    parser.feed(html)
    return " ".join(parser.parts)


def markdown_text(raw: str) -> str:
    """Strip front matter, tables and list/heading markers from a Markdown help-centre page."""
    body = raw.split("---\n", 2)[2] if raw.startswith("---\n") else raw
    lines = []
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith("|"):
            continue
        s = re.sub(r"^(#{1,6}\s+|[-*]\s+|\d+\.\s+|>\s*)", "", s)
        s = s.replace("**", "").replace("`", "")
        lines.append(s)
    return "\n".join(lines)


@dataclass(frozen=True)
class Document:
    """One training document: where it came from and its text."""

    source: str
    text: str


def load_corpus() -> list[Document]:
    """English Hearthwise text: current public help-centre pages plus the web pages a cleaning
    pipeline would keep (``keep: true`` in ``web/labels.jsonl``)."""
    docs: list[Document] = []
    root = datasets_dir()
    for meta in read_jsonl("kb/manifest.jsonl"):
        if meta["format"] != "markdown" or meta["lang"] != "en":
            continue
        if meta["status"] != "current" or meta["audience"] != "public":
            continue
        raw = (root / meta["path"]).read_text(encoding="utf-8")
        docs.append(Document(meta["id"], markdown_text(raw)))
    labels = {row["url"]: row for row in read_jsonl("web/labels.jsonl")}
    for page in read_jsonl("web/pages.jsonl"):
        lab = labels[page["url"]]
        if lab["keep"] and lab["lang"] == "en" and page["content_type"] == "text/html":
            docs.append(Document(page["url"], html_text(page["html"])))
    return docs
