"""A small pre-training data pipeline over the offline crawl in ``datasets/web``.

The stages mirror, in miniature, what web-scale pipelines such as RefinedWeb, Dolma and FineWeb
describe: fetch filtering, URL normalisation and filtering, HTML-to-text extraction, language
identification, heuristic quality filters, exact and near-duplicate removal, benchmark
decontamination and personal-data redaction. Every decision is recorded so you can measure it
against the ground truth in ``datasets/web/labels.jsonl``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .data import read_jsonl
from .dedup import content_hash, near_duplicate_groups

# --------------------------------------------------------------------------- #
# 1. URLs                                                                     #
# --------------------------------------------------------------------------- #

TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid")
URL_BLOCKLIST_WORDS = ("casino", "porn", "viagra", "torrent")  # real lists hold thousands of entries


def normalise_url(url: str) -> str:
    """Canonical form of a URL: lowercase host, no fragment, no tracking parameters, no trailing slash."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if parts.port and not ((parts.scheme == "http" and parts.port == 80) or
                           (parts.scheme == "https" and parts.port == 443)):
        host = f"{host}:{parts.port}"
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith(TRACKING_PARAMS)]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(sorted(query)), ""))


def url_is_blocked(url: str, words: Sequence[str] = URL_BLOCKLIST_WORDS) -> bool:
    """True if the URL contains a blocklisted word (the cheapest filter, applied before download)."""
    low = url.lower()
    return any(w in low for w in words)


# --------------------------------------------------------------------------- #
# 2. HTML to text                                                             #
# --------------------------------------------------------------------------- #

SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "button", "form", "head"}
SKIP_CLASSES = {"cookie-banner", "ads", "byline", "related", "logo"}
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "article", "section", "tr", "br"}
VOID_TAGS = {"br", "meta", "img", "hr", "input", "link", "source", "wbr"}


class _MainTextParser(HTMLParser):
    """Collect visible text, skipping boilerplate elements; prefer what is inside <article>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []  # (tag, starts_a_skipped_region)
        self.skip_depth = 0
        self.article_depth = 0
        self.all_text: list[str] = []
        self.article_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = set((dict(attrs).get("class") or "").split())
        if tag in BLOCK_TAGS:
            self._emit("\n")
        if tag in VOID_TAGS:
            return
        skip = tag in SKIP_TAGS or bool(classes & SKIP_CLASSES)
        self.stack.append((tag, skip))
        self.skip_depth += skip
        self.article_depth += tag == "article"

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        while self.stack:
            open_tag, skip = self.stack.pop()
            self.skip_depth -= skip
            self.article_depth -= open_tag == "article"
            if open_tag == tag:
                break
        if tag in BLOCK_TAGS:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self._emit(data)

    def _emit(self, text: str) -> None:
        self.all_text.append(text)
        if self.article_depth > 0:
            self.article_text.append(text)


def html_to_text(html: str) -> str:
    """Extract the main text of a page: one line per block, boilerplate removed."""
    parser = _MainTextParser()
    parser.feed(html)
    parser.close()
    raw = "".join(parser.article_text) or "".join(parser.all_text)
    lines = (" ".join(line.split()) for line in raw.splitlines())
    return "\n".join(line for line in lines if line)


def looks_binary(text: str) -> bool:
    """Heuristic: control characters or PDF/JPEG magic bytes mean this is not a text page."""
    head = text[:200]
    if head.startswith(("%PDF", "\xff\xd8", "ÿØÿ")):
        return True
    controls = sum(1 for ch in head if ord(ch) < 32 and ch not in "\n\r\t")
    return controls > 0


# --------------------------------------------------------------------------- #
# 3. Language identification                                                  #
# --------------------------------------------------------------------------- #

STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and of to a in is it you that for on with are this your be at or if not can "
                    "from have by when do so no".split()),
    "es": frozenset("el la de que y en los las un una por con para es del se no tu al lo como más cada "
                    "su ya hay".split()),
    "de": frozenset("der die das und ist nicht mit sie ihr den im zu ein eine für auf von dem des sich "
                    "oder ihre ins".split()),
    "fr": frozenset("le la les de des et est un une pour que qui dans en du au par sur pas vos votre "
                    "ce il quand".split()),
}
_LID_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def language_scores(text: str) -> dict[str, float]:
    """Share of words that are stop words of each language."""
    words = [w.lower() for w in _LID_WORD_RE.findall(text)]
    if not words:
        return {lang: 0.0 for lang in STOPWORDS}
    return {lang: sum(w in sw for w in words) / len(words) for lang, sw in STOPWORDS.items()}


def detect_language(text: str, min_score: float = 0.05) -> str:
    """Return the best-scoring language code, or ``"unknown"`` when nothing scores high enough."""
    scores = language_scores(text)
    best = max(scores, key=lambda k: (scores[k], k == "en"))
    return best if scores[best] >= min_score else "unknown"


# --------------------------------------------------------------------------- #
# 4. Quality heuristics                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for the heuristic filters. Change them and watch precision and recall move."""

    min_words: int = 5
    max_top_trigram_share: float = 0.20
    max_caps_ratio: float = 0.25
    max_exclamations_per_sentence: float = 0.5
    min_mean_word_len: float = 3.0
    max_mean_word_len: float = 10.0


def quality_signals(text: str) -> dict[str, float]:
    """Cheap, explainable signals computed from the extracted text."""
    words = _LID_WORD_RE.findall(text)
    n = len(words)
    letters = [ch for ch in text if ch.isalpha()]
    caps = sum(ch.isupper() for ch in letters) / len(letters) if letters else 0.0
    lowered = [w.lower() for w in words]
    trigrams = Counter(tuple(lowered[i : i + 3]) for i in range(max(0, n - 2)))
    top_count = trigrams.most_common(1)[0][1] if trigrams else 0
    top = top_count * 3 / n if top_count >= 2 else 0.0  # only a repeated trigram counts
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    return {
        "words": float(n),
        "mean_word_len": sum(map(len, words)) / n if n else 0.0,
        "caps_ratio": caps,
        "top_trigram_share": top,
        "exclamations_per_sentence": text.count("!") / sentences,
    }


def quality_failures(text: str, cfg: QualityConfig = QualityConfig()) -> list[str]:
    """Names of the quality rules this text breaks (empty list means it passes)."""
    s = quality_signals(text)
    fails = []
    if s["words"] < cfg.min_words:
        fails.append("too_short")
    if s["top_trigram_share"] > cfg.max_top_trigram_share:
        fails.append("repetitive")
    if s["caps_ratio"] > cfg.max_caps_ratio:
        fails.append("shouting")
    if s["exclamations_per_sentence"] > cfg.max_exclamations_per_sentence:
        fails.append("exclamations")
    if s["words"] and not cfg.min_mean_word_len <= s["mean_word_len"] <= cfg.max_mean_word_len:
        fails.append("odd_word_length")
    return fails


# --------------------------------------------------------------------------- #
# 5. Decontamination                                                          #
# --------------------------------------------------------------------------- #

_NGRAM_WORD_RE = re.compile(r"\w+", re.UNICODE)


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = [w.lower() for w in _NGRAM_WORD_RE.findall(text)]
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def eval_texts() -> list[str]:
    """Texts we will evaluate on later in the course: they must never be trained on."""
    texts = [f"{r['question']} {r['reference_answer']}" for r in read_jsonl("rag_eval/qa.jsonl")]
    texts += [r["question"] for r in read_jsonl("reasoning/problems.jsonl")]
    return texts


def build_eval_index(texts: Iterable[str], n: int = 8) -> set[tuple[str, ...]]:
    index: set[tuple[str, ...]] = set()
    for t in texts:
        index |= word_ngrams(t, n)
    return index


def contamination_hits(text: str, index: set[tuple[str, ...]], n: int = 8) -> int:
    """How many of the document's word n-grams also occur in an evaluation item."""
    return len(word_ngrams(text, n) & index)


# --------------------------------------------------------------------------- #
# 6. Personal data                                                            #
# --------------------------------------------------------------------------- #

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "PHONE": re.compile(r"\+\d{1,3}(?:[ -]?\d{2,4}){2,4}"),
    "ADDRESS": re.compile(r"\b(?:Calle|Rua|Avenida|Street|Road)\s+[A-ZÁÉÍÓÚ]\w+(?:\s[A-Z]\w+)*\s+\d{1,4}"
                          r"(?:,\s*[A-Z]\w+)?"),
}


def redact_pii(text: str) -> tuple[str, Counter[str]]:
    """Replace e-mails, phone numbers and street addresses with placeholders like ``<EMAIL>``."""
    found: Counter[str] = Counter()
    for kind, pattern in PII_PATTERNS.items():
        text, n = pattern.subn(f"<{kind}>", text)
        found[kind] += n
    return text, +found


# --------------------------------------------------------------------------- #
# 7. The pipeline                                                             #
# --------------------------------------------------------------------------- #

KEEP_LANGS = ("en", "es")  # Hearthwise serves customers in English and Spanish


@dataclass
class Doc:
    """One crawled page and every decision the pipeline made about it."""

    url: str
    status: int
    content_type: str
    html: str
    norm_url: str = ""
    text: str = ""
    lang: str = ""
    dropped: str = ""  # empty while the document survives
    pii: Counter[str] = field(default_factory=Counter)

    @property
    def kept(self) -> bool:
        return not self.dropped


@dataclass(frozen=True)
class PipelineConfig:
    keep_langs: tuple[str, ...] = KEEP_LANGS
    quality: QualityConfig = QualityConfig()
    near_dup_threshold: float = 0.8
    contamination_ngram: int = 8
    contamination_min_hits: int = 1
    use_url_dedup: bool = True
    use_exact_dedup: bool = True
    use_near_dedup: bool = True
    use_decontamination: bool = True


STAGES = ["fetch", "url_dedup", "url_filter", "extract", "language", "quality", "exact_dup",
          "near_dup", "contamination"]


def load_crawl() -> list[Doc]:
    return [Doc(url=r["url"], status=r["status"], content_type=r["content_type"], html=r["html"])
            for r in read_jsonl("web/pages.jsonl")]


def _keeper(docs: Sequence[Doc]) -> Doc:
    """Which copy survives in a duplicate group: the shortest URL (often the original), then alphabetical."""
    return min(docs, key=lambda d: (len(d.url), d.url))


def run_pipeline(docs: list[Doc], cfg: PipelineConfig = PipelineConfig()) -> list[Doc]:
    """Run every stage in order. Each document ends with ``dropped`` empty (kept) or set to a reason."""
    alive = lambda: [d for d in docs if d.kept]  # noqa: E731

    for d in docs:  # fetch: only successful HTML responses carry text
        if d.status != 200:
            d.dropped = "fetch:http_error"
        elif d.content_type != "text/html" or looks_binary(d.html):
            d.dropped = "fetch:not_html"

    for d in docs:
        d.norm_url = normalise_url(d.url)
    if cfg.use_url_dedup:
        groups: dict[str, list[Doc]] = {}
        for d in alive():
            groups.setdefault(d.norm_url, []).append(d)
        for group in groups.values():
            keep = _keeper(group)
            for d in group:
                if d is not keep:
                    d.dropped = "url_dedup"

    for d in alive():
        if url_is_blocked(d.url):
            d.dropped = "url_filter"

    for d in alive():
        d.text = html_to_text(d.html)
        if not d.text:
            d.dropped = "extract:empty"

    for d in alive():
        d.lang = detect_language(d.text)
        if d.lang not in cfg.keep_langs:
            d.dropped = f"language:{d.lang}"

    for d in alive():
        fails = quality_failures(d.text, cfg.quality)
        if fails:
            d.dropped = f"quality:{fails[0]}"

    if cfg.use_exact_dedup:
        by_hash: dict[str, list[Doc]] = {}
        for d in alive():
            by_hash.setdefault(content_hash(d.text), []).append(d)
        for group in by_hash.values():
            keep = _keeper(group)
            for d in group:
                if d is not keep:
                    d.dropped = "exact_dup"

    if cfg.use_near_dedup:
        survivors = alive()
        for group in near_duplicate_groups([d.text for d in survivors], threshold=cfg.near_dup_threshold):
            members = [survivors[i] for i in group]
            keep = _keeper(members)
            for d in members:
                if d is not keep:
                    d.dropped = "near_dup"

    if cfg.use_decontamination:
        index = build_eval_index(eval_texts(), cfg.contamination_ngram)
        for d in alive():
            if contamination_hits(d.text, index, cfg.contamination_ngram) >= cfg.contamination_min_hits:
                d.dropped = "contamination"

    for d in alive():  # PII is redacted, not dropped: the rest of the page is still useful text
        d.text, d.pii = redact_pii(d.text)
    return docs


# --------------------------------------------------------------------------- #
# 8. Measuring the pipeline                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Report:
    total: int
    kept: int
    tp: int
    fp: int
    fn: int
    tn: int
    dropped_by_stage: dict[str, int]
    html_chars: int
    text_chars_kept: int
    pii_pages: int
    pii_counts: dict[str, int]
    false_keeps: list[str]
    false_drops: list[tuple[str, str]]

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(docs: Sequence[Doc], labels: Sequence[dict[str, Any]]) -> Report:
    """Compare keep/drop decisions with the ground-truth ``keep`` label ("keep" is the positive class)."""
    truth = {row["url"]: bool(row["keep"]) for row in labels}
    tp = fp = fn = tn = 0
    false_keeps: list[str] = []
    false_drops: list[tuple[str, str]] = []
    for d in docs:
        gold = truth[d.url]
        if d.kept and gold:
            tp += 1
        elif d.kept:
            fp += 1
            false_keeps.append(d.url)
        elif gold:
            fn += 1
            false_drops.append((d.url, d.dropped))
        else:
            tn += 1
    stages = Counter(d.dropped.split(":")[0] for d in docs if not d.kept)
    pii_counts: Counter[str] = Counter()
    for d in docs:
        pii_counts.update(d.pii)
    return Report(
        total=len(docs),
        kept=sum(d.kept for d in docs),
        tp=tp, fp=fp, fn=fn, tn=tn,
        dropped_by_stage={s: stages.get(s, 0) for s in STAGES},
        html_chars=sum(len(d.html) for d in docs),
        text_chars_kept=sum(len(d.text) for d in docs if d.kept),
        pii_pages=sum(1 for d in docs if d.kept and d.pii),
        pii_counts=dict(sorted(pii_counts.items())),
        false_keeps=sorted(false_keeps),
        false_drops=sorted(false_drops),
    )


def clean_crawl(cfg: PipelineConfig = PipelineConfig()) -> tuple[list[Doc], Report]:
    """Load the crawl, run the pipeline and evaluate it against the labels."""
    docs = run_pipeline(load_crawl(), cfg)
    return docs, evaluate(docs, read_jsonl("web/labels.jsonl"))


def kept_texts(docs: Sequence[Doc], lang: str | None = None) -> list[str]:
    """Clean text of the surviving documents (optionally one language), in URL order."""
    return [d.text for d in sorted(docs, key=lambda d: d.url) if d.kept and (lang is None or d.lang == lang)]
