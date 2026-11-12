"""Read the Hearthwise help centre (datasets/kb) for the lab's tools.

- :func:`load_docs` parses Markdown front matter and HTML ``<meta>`` tags and
  turns HTML into plain text, dropping navigation, footers and **hidden
  elements** (``display:none``), where indirect prompt injections like to hide.
- :func:`get_product` reads a product page's specification table.
- :func:`search` is a small TF-IDF keyword search that, by default, skips
  internal and superseded documents (access control and freshness *before*
  the model ever sees the text).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from html.parser import HTMLParser

from .data import datasets_dir, read_jsonl

PRODUCT_SLUGS = ["ember-t3", "glow-a19", "spark-mini", "watcher-s1", "lookout-c2", "hearth-hub"]

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it its my of on or our the to "
    "what when where which who why will with you your long take get de la el en es los las por que un una y".split()
)


@dataclass(frozen=True)
class Doc:
    """One help-centre document with its metadata and plain text."""

    id: str
    title: str
    path: str
    lang: str
    doc_type: str
    status: str
    audience: str
    source: str
    effective_date: str
    text: str
    hidden_removed: int = 0
    paragraphs: tuple[str, ...] = field(default=(), repr=False)


class _TextExtractor(HTMLParser):
    """HTML to text that skips nav/footer/script/style and anything hidden with CSS."""

    SKIP_TAGS = {"nav", "footer", "script", "style", "head"}
    BLOCK_TAGS = {"p", "h1", "h2", "h3", "li", "tr", "table", "div", "main"}
    VOID_TAGS = {"meta", "br", "img", "input", "link", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.hidden_removed = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.VOID_TAGS:
            return
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        hidden = "display:none" in style or "visibility:hidden" in style or ("hidden", None) in attrs
        if self.skip_depth or tag in self.SKIP_TAGS or hidden:
            if hidden and not self.skip_depth:
                self.hidden_removed += 1
            self.skip_depth += 1
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID_TAGS:
            return
        if self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, int]:
    """Return (visible text, number of hidden elements removed)."""
    parser = _TextExtractor()
    parser.feed(html)
    text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text, parser.hidden_removed


def _strip_front_matter(md: str) -> str:
    if md.startswith("---"):
        end = md.find("\n---", 3)
        if end != -1:
            return md[end + 4 :].lstrip("\n")
    return md


def _paragraphs(text: str) -> tuple[str, ...]:
    """Split into paragraphs; a heading is glued to the paragraph that follows it."""
    out: list[str] = []
    pending = ""
    for block in re.split(r"\n\s*\n", text):
        flat = " ".join(block.replace("**", "").split())
        if not flat:
            continue
        if flat.startswith("#") and "\n" not in block.strip():
            pending = f"{pending} {flat.lstrip('# ')}:".strip()
            continue
        out.append(f"{pending} {flat}".strip())
        pending = ""
    if pending:
        out.append(pending)
    return tuple(out)


@lru_cache(maxsize=1)
def load_docs() -> tuple[Doc, ...]:
    """Load every document listed in ``kb/manifest.jsonl`` (cached)."""
    base = datasets_dir()
    docs: list[Doc] = []
    for meta in read_jsonl("kb/manifest.jsonl"):
        raw = (base / meta["path"]).read_text(encoding="utf-8")
        hidden = 0
        if meta["format"] == "html":
            text, hidden = html_to_text(raw)
        else:
            text = _strip_front_matter(raw)
        docs.append(
            Doc(
                id=meta["id"], title=meta["title"], path=meta["path"], lang=meta["lang"],
                doc_type=meta["doc_type"], status=meta["status"], audience=meta["audience"],
                source=meta["source"], effective_date=meta["effective_date"], text=text,
                hidden_removed=hidden, paragraphs=_paragraphs(text),
            )
        )
    return tuple(docs)


def tokenize(text: str) -> list[str]:
    """Lowercase words without stop words."""
    return [w for w in (t.lower() for t in _WORD_RE.findall(text)) if w not in _STOPWORDS]


def searchable_docs(include_internal: bool = False, include_superseded: bool = False) -> list[Doc]:
    """Documents a customer-facing tool may read."""
    return [
        d
        for d in load_docs()
        if (include_internal or d.audience == "public") and (include_superseded or d.status == "current")
    ]


def search(query: str, top_k: int = 3, **filters: bool) -> list[dict[str, str | float]]:
    """TF-IDF keyword search. Returns id, title, source, effective date, score and best snippet."""
    docs = searchable_docs(**filters)
    q_terms = set(tokenize(query))
    df = Counter(t for d in docs for t in set(tokenize(d.title + " " + d.text)))
    n = len(docs)
    scored: list[tuple[float, Doc]] = []
    for d in docs:
        tf = Counter(tokenize(d.title + " " + d.title + " " + d.text))  # title counts twice
        score = sum(math.log(1 + tf[t]) * math.log((n + 1) / (df[t] + 0.5)) for t in q_terms if tf[t])
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    results = []
    for score, d in scored[:top_k]:
        best = max(d.paragraphs, key=lambda p: (len(q_terms & set(tokenize(p))), -d.paragraphs.index(p)))
        results.append({
            "id": d.id, "title": d.title, "source": d.source, "effective_date": d.effective_date,
            "score": round(score, 3), "snippet": best[:240],
        })
    return results


def _table_rows(md: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] not in ("Spec", "") and not set(cells[0]) <= {"-"}:
            rows[cells[0]] = cells[1]
    return rows


@lru_cache(maxsize=None)
def get_product(slug: str) -> dict[str, object]:
    """Product facts parsed from ``kb/products/<slug>.md`` (the catalogue as published)."""
    if slug not in PRODUCT_SLUGS:
        raise KeyError(slug)
    doc = next(d for d in load_docs() if d.path == f"kb/products/{slug}.md")
    rows = _table_rows(doc.text)
    features = re.findall(r"^- (.+)$", doc.text.split("## Key features", 1)[1].split("##", 1)[0], re.M)
    words = doc.title.split()  # e.g. "Ember T3 smart thermostat": name, then category
    return {
        "name": " ".join(words[:2]),
        "category": " ".join(words[2:]),
        "sku": rows["SKU"],
        "price_eur": float(rows["Price"].lstrip("€")),
        "wifi": rows["Wi-Fi"],
        "standby_w": float(rows["Standby power"].removesuffix(" W")),
        "power": rows["Power"],
        "needs_hub": rows["Requires Hearth Hub"] == "Yes",
        "launched": rows["Launched"],
        "warranty": rows["Warranty"],
        "features": features,
        "source_doc": doc.id,
    }
