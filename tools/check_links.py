#!/usr/bin/env python3
"""Check links in the course Markdown.

Default (offline, used in CI):
- relative links point to files that exist;
- ``#anchors`` exist in the target page (GitHub slug rules);
- a Spanish page links to the Spanish version of a page when one exists;
- no Markdown link points to a fictional ``.example`` domain.

With ``--external`` it also requests every http(s) URL and reports the ones
that fail. Some sites refuse automated requests (403/429); those are listed as
"unverified" rather than broken.

Usage::

    python3 tools/check_links.py [paths...]
    python3 tools/check_links.py --external
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _md import REPO, anchors, is_spanish, markdown_files, parse, rel, sibling  # noqa: E402

LINK_RE = re.compile(r"(?<!!)\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
IMG_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?\s*\)")
AUTO_RE = re.compile(r"<(https?://[^>\s]+)>")
BARE_RE = re.compile(r"(?<![(<\"\[])\bhttps?://[^\s)<>|`\"*]+")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
USER_AGENT = "Mozilla/5.0 (compatible; course-link-checker/1.0)"


def links_in(path: Path) -> list[tuple[int, str]]:
    parsed = parse(path.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for lineno, line in parsed.prose_lines:
        clean = INLINE_CODE_RE.sub("", line)
        for rx in (LINK_RE, IMG_RE, AUTO_RE):
            out.extend((lineno, m.group(1)) for m in rx.finditer(clean))
        for m in BARE_RE.finditer(clean):  # GitHub auto-links bare URLs too
            url = m.group(0).rstrip(".,;:")
            if not any(url == t for _, t in out):
                out.append((lineno, url))
    return out


_anchor_cache: dict[Path, set[str]] = {}


def anchors_of(path: Path) -> set[str]:
    if path not in _anchor_cache:
        _anchor_cache[path] = anchors(parse(path.read_text(encoding="utf-8")))
    return _anchor_cache[path]


def check_internal(path: Path, lineno: int, target: str) -> str | None:
    if target.startswith(("mailto:", "tel:")):
        return None
    if target.startswith("#"):
        frag = urllib.parse.unquote(target[1:])
        return None if frag in anchors_of(path) else f"anchor #{frag} not found in this page"
    raw, _, frag = target.partition("#")
    raw = urllib.parse.unquote(raw)
    dest = (path.parent / raw).resolve() if not raw.startswith("/") else (REPO / raw.lstrip("/")).resolve()
    if not dest.exists():
        return f"broken link: {target}"
    try:
        dest.relative_to(REPO)
    except ValueError:
        return f"link leaves the repository: {target}"
    if is_spanish(path) and dest.suffix == ".md" and not is_spanish(dest) and lineno > 1:
        es = sibling(dest)
        if es.exists():
            return f"Spanish page should link to {es.name} instead of {dest.name}"
    if frag and dest.suffix == ".md" and urllib.parse.unquote(frag) not in anchors_of(dest):
        return f"anchor #{frag} not found in {rel(dest)}"
    return None


def fetch(url: str) -> tuple[str, str]:
    """Return (status, detail) where status is ok | broken | unverified."""
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                return "ok", str(resp.status)
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code in (403, 404, 405, 429, 500, 501, 503):
                continue
            if exc.code in (401, 403, 429, 503):
                return "unverified", f"HTTP {exc.code}"
            return "broken", f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001 - report any network failure
            if method == "HEAD":
                continue
            return "unverified", type(exc).__name__
    return "unverified", "no response"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="files or folders (default: whole repository)")
    ap.add_argument("--external", action="store_true", help="also request every http(s) URL")
    args = ap.parse_args()

    errors: list[str] = []
    external: dict[str, list[str]] = {}
    files = markdown_files(args.paths or None)
    for path in files:
        for lineno, target in links_in(path):
            where = f"{rel(path)}:{lineno}"
            if target.startswith(("http://", "https://")):
                host = urllib.parse.urlparse(target).hostname or ""
                if host.endswith(".example"):
                    errors.append(f"{where}: links to a fictional domain ({target}); write it as plain text")
                    continue
                external.setdefault(target, []).append(where)
                continue
            problem = check_internal(path, lineno, target)
            if problem:
                errors.append(f"{where}: {problem}")

    for e in errors:
        print(f"❌ {e}")
    print(f"\nChecked {len(files)} pages: {len(errors)} internal problem(s), {len(external)} unique external URLs.")

    if not args.external:
        return 1 if errors else 0

    broken = unverified = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(zip(external, pool.map(fetch, external)))
    for url, (status, detail) in sorted(results.items()):
        if status == "ok":
            continue
        mark = "❌" if status == "broken" else "⚠️ "
        broken += status == "broken"
        unverified += status == "unverified"
        print(f"{mark} {status} ({detail}): {url}  ← {external[url][0]}")
    print(f"External: {len(external) - broken - unverified} ok, {broken} broken, {unverified} unverified.")
    return 1 if errors or broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
