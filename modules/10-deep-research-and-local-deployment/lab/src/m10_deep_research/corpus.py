"""Load the offline web snapshot (``datasets/web``) as clean, searchable pages.

The snapshot is a raw crawl: HTML with navigation, cookie banners, footers,
error pages, binary files and tracking-parameter duplicates. This module turns
it into :class:`Page` objects that hold only the article text.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .data import read_jsonl

TRACKING_PREFIXES = ("utm_",)


def canonical_url(url: str) -> str:
    """Drop tracking parameters and fragments so copies of one page share one URL."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.startswith(TRACKING_PREFIXES)]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, urlencode(query), ""))


class _ArticleParser(HTMLParser):
    """Collect <title>, <html lang>, <meta name=date> and the <p> elements inside <article>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.lang = ""
        self.published = ""
        self.paragraphs: list[str] = []
        self._in_title = False
        self._in_article = 0
        self._in_p = False
        self._skip_p = False
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if tag == "html":
            self.lang = a.get("lang", "")
        elif tag == "title":
            self._in_title = True
        elif tag == "meta" and a.get("name") == "date":
            self.published = a.get("content", "")
        elif tag == "article":
            self._in_article += 1
        elif tag == "p" and self._in_article:
            self._in_p, self._skip_p, self._buf = True, a.get("class") == "byline", []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "article":
            self._in_article = max(0, self._in_article - 1)
        elif tag == "p" and self._in_p:
            text = " ".join("".join(self._buf).split())
            if text and not self._skip_p:
                self.paragraphs.append(text)
            self._in_p = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._in_p:
            self._buf.append(data)


@dataclass(frozen=True)
class Page:
    """One crawled page, reduced to what a research agent needs."""

    url: str
    canonical: str
    domain: str
    status: int
    content_type: str
    title: str
    publisher: str
    published: str
    lang: str
    paragraphs: tuple[str, ...]
    fetched: str = ""

    @property
    def ok(self) -> bool:
        """True for a 200 HTML page with at least one paragraph of article text."""
        return self.status == 200 and self.content_type == "text/html" and bool(self.paragraphs)

    @property
    def text(self) -> str:
        return " ".join(self.paragraphs)

    @property
    def published_date(self) -> date | None:
        try:
            return date.fromisoformat(self.published)
        except ValueError:
            return None


def parse_page(record: dict) -> Page:
    """Turn one raw crawl record into a :class:`Page`."""
    url = record["url"]
    domain = urlsplit(url).netloc.lower()
    status, ctype = int(record["status"]), record["content_type"]
    if ctype != "text/html":
        fetched = record["fetched_at"][:10]
        return Page(url, canonical_url(url), domain, status, ctype, "", "", "", "", (), fetched)
    parser = _ArticleParser()
    parser.feed(record["html"])
    full_title = html.unescape(parser.title).strip()
    title, _, publisher = full_title.partition(" | ")
    return Page(
        url=url,
        canonical=canonical_url(url),
        domain=domain,
        status=status,
        content_type=ctype,
        title=title.strip(),
        publisher=publisher.strip(),
        published=parser.published,
        lang=parser.lang,
        paragraphs=tuple(parser.paragraphs),
        fetched=record["fetched_at"][:10],
    )


class Snapshot:
    """All pages of the offline web, addressable by URL (raw or canonical)."""

    def __init__(self, pages: list[Page]) -> None:
        self.pages = pages
        self._by_url = {p.url: p for p in pages}
        for p in pages:  # the first page seen for a canonical URL wins
            self._by_url.setdefault(p.canonical, p)

    def get(self, url: str) -> Page | None:
        return self._by_url.get(url) or self._by_url.get(canonical_url(url))

    def searchable(self) -> list[Page]:
        """Pages worth indexing: successful HTML with article text."""
        return [p for p in self.pages if p.ok]

    @property
    def as_of(self) -> date:
        """The snapshot date: the day the last page was crawled (used for recency)."""
        return max(date.fromisoformat(p.fetched) for p in self.pages)


def load_snapshot() -> Snapshot:
    """Read ``datasets/web/pages.jsonl`` into a :class:`Snapshot`."""
    return Snapshot([parse_page(r) for r in read_jsonl("web/pages.jsonl")])


def load_labels() -> dict[str, dict]:
    """Ground-truth labels by URL. Only the grader may read these; the agent never does."""
    return {row["url"]: row for row in read_jsonl("web/labels.jsonl")}


def load_questions() -> dict[str, dict]:
    """The deep-research questions (``rq1`` ... ``rq4``) by id."""
    return {row["id"]: row for row in read_jsonl("agent_tasks/research_questions.jsonl")}
