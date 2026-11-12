"""Load the offline web snapshot and turn raw HTML into clean article text.

The snapshot (``datasets/web/pages.jsonl``) is our stand-in for the live web:
the same request always returns the same page, so every run is reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .data import read_jsonl

SKIP_TAGS = {"head", "header", "nav", "footer", "aside", "script", "style"}
BLOCK_TAGS = {"p", "h1", "h2", "h3", "li", "td", "th", "tr", "div", "article", "section"}
VOID_TAGS = {"br", "meta", "img", "input", "hr", "link"}


@dataclass(frozen=True)
class Page:
    """One crawled page after extraction."""

    url: str
    canonical: str
    domain: str
    status: int
    content_type: str
    title: str
    published: str  # ISO date (YYYY-MM-DD) or "" when the page has none
    lang: str
    text: str  # visible article text, one block per line

    @property
    def ok(self) -> bool:
        """True for pages an agent can read: HTTP 200 and HTML."""
        return self.status == 200 and self.content_type.startswith("text/html")


class _Extractor(HTMLParser):
    """Keep visible article text; drop navigation, banners, ads and hidden elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[bool] = []
        self.skip_depth = 0
        self.blocks: list[list[str]] = [[]]
        self.title_parts: list[str] = []
        self.in_title = False
        self.meta_date = ""
        self.lang = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if tag == "html":
            self.lang = a.get("lang", "")
        if tag == "meta" and a.get("name") == "date":
            self.meta_date = a.get("content", "")
        if tag == "title":
            self.in_title = True
        if tag in VOID_TAGS:
            return
        cls = a.get("class", "")
        hidden = "display:none" in a.get("style", "").replace(" ", "") or "hidden" in a
        skip = tag in SKIP_TAGS or "cookie" in cls or cls == "ads" or hidden
        self.stack.append(skip)
        self.skip_depth += skip
        if tag in BLOCK_TAGS:
            self.blocks.append([])

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag in VOID_TAGS or not self.stack:
            return
        self.skip_depth -= self.stack.pop()
        if tag in BLOCK_TAGS:
            self.blocks.append([])

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        elif self.skip_depth == 0 and data.strip():
            self.blocks[-1].append(data.strip())


def html_to_text(html: str) -> tuple[str, str, str, str]:
    """Return ``(title, text, published, lang)`` for one HTML document."""
    ex = _Extractor()
    ex.feed(html)
    lines = [" ".join(b) for b in ex.blocks if b]
    title = " ".join("".join(ex.title_parts).split())
    title = title.split(" | ")[0]  # drop the " | Site name" suffix
    return title, "\n".join(lines), ex.meta_date, ex.lang


def canonical_url(url: str) -> str:
    """Drop tracking parameters (``utm_*``) and fragments so copies share one URL."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.startswith("utm_")]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _to_page(row: dict) -> Page:
    url = row["url"]
    title, text, published, lang = ("", "", "", "")
    if row["content_type"].startswith("text/html"):
        title, text, published, lang = html_to_text(row["html"])
    return Page(
        url=url,
        canonical=canonical_url(url),
        domain=urlsplit(url).netloc,
        status=int(row["status"]),
        content_type=row["content_type"],
        title=title,
        published=published if re.fullmatch(r"\d{4}-\d{2}-\d{2}", published) else "",
        lang=lang,
        text=text,
    )


class Snapshot:
    """All crawled pages, addressable by URL."""

    def __init__(self, pages: tuple[Page, ...]) -> None:
        self.pages = pages
        self._by_url = {p.url: p for p in pages}

    def get(self, url: str) -> Page | None:
        return self._by_url.get(url)

    @property
    def readable(self) -> list[Page]:
        """Pages that return HTTP 200 HTML (the ones worth indexing)."""
        return [p for p in self.pages if p.ok]


@lru_cache(maxsize=1)
def load_snapshot() -> Snapshot:
    """Read ``datasets/web/pages.jsonl`` once and extract every page."""
    return Snapshot(tuple(_to_page(r) for r in read_jsonl("web/pages.jsonl")))


@lru_cache(maxsize=1)
def load_labels() -> dict[str, dict]:
    """Ground-truth labels by URL. Used for evaluation only, never by the agent."""
    return {row["url"]: row for row in read_jsonl("web/labels.jsonl")}
