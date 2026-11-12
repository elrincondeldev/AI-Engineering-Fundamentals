"""Rule-based parsing: front matter, Markdown structure, HTML metadata, tables and hidden text."""

from __future__ import annotations

from m05_retrieval.parsing import load_kb, parse_html, parse_markdown_body, split_front_matter

MD = """---
id: kb-999
status: current
---
# Title

Intro line one
continues here.

## Specs

| Spec | Value |
|---|---|
| Weight | 34 g |

- first item
- second item
"""


def test_front_matter_is_split_from_body() -> None:
    meta, body = split_front_matter(MD)
    assert meta == {"id": "kb-999", "status": "current"}
    assert body.startswith("# Title")


def test_markdown_blocks_keep_structure() -> None:
    _, body = split_front_matter(MD)
    blocks = parse_markdown_body(body)
    kinds = [b.kind for b in blocks]
    assert kinds == ["heading", "text", "heading", "table", "text", "text"]
    assert blocks[1].text == "Intro line one continues here."
    assert blocks[3].rows == (("Spec", "Value"), ("Weight", "34 g"))


def test_html_drops_navigation_and_hidden_text() -> None:
    html = """<html><head><title>T</title><meta name="id" content="kb-1"></head><body>
    <nav>Home | Shop</nav><main><h1>Hello</h1><p>Visible.</p>
    <p style="display: none">Secret instructions.</p>
    <table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table></main>
    <footer>Copyright</footer></body></html>"""
    meta, title, blocks, hidden = parse_html(html)
    assert meta == {"id": "kb-1"} and title == "T"
    text = " ".join(b.text for b in blocks)
    assert "Visible." in text and "Secret" not in text and "Home" not in text and "Copyright" not in text
    assert hidden == 1
    assert blocks[-1].rows == (("A", "B"), ("1", "2"))


def test_whole_knowledge_base_parses() -> None:
    docs = load_kb()
    assert len(docs) == 26
    by_id = {d.doc_id: d for d in docs}
    assert by_id["kb-010"].meta["status"] == "superseded"
    assert by_id["kb-041"].meta["audience"] == "internal"
    assert by_id["kb-031"].meta["lang"] == "es"
    assert by_id["kb-025"].hidden_dropped == 1
    assert "verify@" not in by_id["kb-025"].text
    table = next(b for b in by_id["kb-007"].blocks if b.kind == "table")
    assert table.rows[0][5] == "Weight (g)" and table.rows[4][:1] == ("Watcher S1",)
