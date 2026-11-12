"""Load the Hearthwise help centre (datasets/kb) into documents and heading-level chunks.

Metadata comes from ``kb/manifest.jsonl``: we filter on it before retrieval, and
we show it to the model inside the prompt. Bodies come from the Markdown or HTML
files; tables are linearised into one sentence per row so that facts hidden in
tables can be retrieved and quoted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .data import datasets_dir, read_jsonl
from .sanitize import html_to_text, quarantine
from .text import normalize


@dataclass(frozen=True)
class Document:
    """One help-centre page with its metadata and cleaned text."""

    id: str
    path: str
    title: str
    lang: str
    doc_type: str
    effective_date: str
    status: str
    audience: str
    source: str
    text: str
    removed: tuple[str, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class Chunk:
    """A heading-level piece of a document. ``doc`` carries the metadata."""

    chunk_id: str
    doc: Document
    section: str
    text: str

    @property
    def doc_id(self) -> str:
        return self.doc.id


def _linearise_markdown_tables(body: str) -> str:
    """Rewrite '| a | b |' tables as one 'Header: value; Header: value.' line per row."""
    out: list[str] = []
    header: list[str] | None = None
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if header is None:
                header = cells
            elif all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                continue
            else:
                out.append(linearise_row(header, cells))
                out.append("")
            continue
        header = None
        out.append(line)
    return "\n".join(out)


def linearise_row(header: list[str], cells: list[str]) -> str:
    """One table row as one sentence: 'Code: E2; Meaning: ...; What to do: ...'."""
    row = "; ".join(f"{h}: {v}" for h, v in zip(header, cells)).rstrip()
    return row if row.endswith(".") else row + "."


def _strip_front_matter(raw: str) -> str:
    if raw.startswith("---\n"):
        return raw.split("---\n", 2)[2]
    return raw


def _clean_inline(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)


def load_documents(*, sanitize: bool = True) -> list[Document]:
    """Read every document listed in the manifest.

    ``sanitize=True`` drops hidden HTML and quarantines instruction-like
    paragraphs. ``sanitize=False`` is the naive pipeline used in the ablation.
    """
    return list(_load_documents_cached(sanitize))


@lru_cache(maxsize=2)
def _load_documents_cached(sanitize: bool) -> tuple[Document, ...]:
    root = datasets_dir()
    docs: list[Document] = []
    for meta in read_jsonl("kb/manifest.jsonl"):
        raw = (root / meta["path"]).read_text(encoding="utf-8")
        removed: list[str] = []
        if meta["format"] == "markdown":
            text = _linearise_markdown_tables(_strip_front_matter(raw))
        else:
            extracted = html_to_text(raw, drop_hidden=sanitize)
            text = extracted.text
            removed.extend(extracted.hidden if sanitize else [])
        text = _clean_inline(text)
        if sanitize:
            text, flagged = quarantine(text)
            removed.extend(flagged)
        docs.append(
            Document(
                id=meta["id"], path=meta["path"], title=meta["title"], lang=meta["lang"],
                doc_type=meta["doc_type"], effective_date=meta["effective_date"], status=meta["status"],
                audience=meta["audience"], source=meta["source"], text=text.strip(), removed=tuple(removed),
            )
        )
    return tuple(docs)


def chunk_document(doc: Document) -> list[Chunk]:
    """Split at Markdown headings (#, ##, ###). Each chunk starts with its heading line."""
    chunks: list[Chunk] = []
    section, lines = doc.title, []

    def emit() -> None:
        body = "\n".join(lines).strip()
        has_body = any(ln.strip() and not ln.lstrip().startswith("#") for ln in lines)
        if has_body:
            chunks.append(Chunk(f"{doc.id}#{len(chunks) + 1}", doc, section, body))

    for line in doc.text.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)$", line.strip())
        if m:
            emit()
            section, lines = m.group(2).strip(), [line.strip()]
        else:
            lines.append(line)
    emit()
    return chunks


def build_chunks(docs: list[Document], *, dedupe: bool = True) -> list[Chunk]:
    """Chunk every document. With ``dedupe`` an identical chunk body is kept only once."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for doc in docs:
        for chunk in chunk_document(doc):
            key = normalize(chunk.text)
            if dedupe and key in seen:
                continue
            seen.add(key)
            out.append(chunk)
    return out
