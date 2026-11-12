"""Text helpers shared by the retriever, the stand-in model and the metrics.

Everything here is deliberately simple and deterministic: lowercase, strip
accents, split into word tokens, drop function words and apply a tiny suffix
stripper. Real systems use proper tokenizers and language models; these helpers
are enough to make the ideas visible.
"""

from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,:][0-9]+)*")
PLACEHOLDER_RE = re.compile(r"\[(?:EMAIL|PHONE|CARD)\]")  # inserted by guardrails.redact_pii
NUMBER_RE = re.compile(r"\d+(?:[.,:]\d+)*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9¿¡\"(])")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
TABLE_ROW_RE = re.compile(r"^[^;:]{1,40}: [^;]*(?:; [^;:]{1,40}: [^;]*)+$")

EN_STOP = frozenset(
    """a about after all also am an and any are as at be been before but by can could did do does
    doing don each else even ever for from get gets got had has have how i if in into is it its
    just long many me more most much my no not of off on once only or other our out over own per
    should so some such than that the their them then there these they this those through to too under
    until up us very was we were what when where which while who whom why will with within would
    you your yours any hi hello hey please thanks thank m s t d ll ve re""".split()
)
ES_STOP = frozenset(
    """a al algo como con cual cuales cuando cuanto cuanta cuantos cuantas de del desde donde
    dura el ella ellas ellos en entre era es esa ese eso esta este esto estos hay la las le les lo
    los mas me mi mis muy no nos o para pero por que quien se si sin sobre su sus te tengo ti tu
    tus un una uno unos unas y ya yo hola gracias favor""".split()
)
STOPWORDS = EN_STOP | ES_STOP

# Words that are frequent in one language and rare in the other: used only for language detection.
_EN_MARKERS = frozenset("the is are what how does do my can will which you your and of to with".split())
_ES_MARKERS = frozenset(
    "el la los las que como cuanto cuantos cuanta es mi mis puedo tengo para con de y".split()
)


def strip_accents(text: str) -> str:
    """Remove diacritics: 'días' -> 'dias', 'México' -> 'Mexico'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """Lowercase, strip accents and Markdown emphasis, collapse whitespace."""
    text = strip_accents(text.replace("**", "").replace("`", "")).lower()
    return re.sub(r"\s+", " ", text).strip()


def stem(token: str) -> str:
    """A tiny suffix stripper: 'devices' -> 'device', 'stored' -> 'stor', 'store' -> 'stor'.

    It only needs to map the forms of one word to the same key; the key does not
    have to be a real word.
    """
    if len(token) < 4 or token[0].isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            if len(token) > 3 and token[-1] == token[-2] and token[-1] not in "lsz":
                token = token[:-1]  # 'swapped' -> 'swapp' -> 'swap'
            break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def tokens(text: str) -> list[str]:
    """All normalised word tokens, including function words. Redaction placeholders are skipped."""
    return TOKEN_RE.findall(normalize(PLACEHOLDER_RE.sub(" ", text)))


def content_tokens(text: str) -> list[str]:
    """Normalised, stemmed tokens without function words (order kept, duplicates kept)."""
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


def numbers(text: str) -> set[str]:
    """Every number in the text, normalised ('€4.99' -> '4.99', '03:00' -> '03:00')."""
    return set(NUMBER_RE.findall(strip_accents(text)))


def split_units(text: str) -> list[str]:
    """Split a chunk into answer-sized units: sentences, list items and table rows.

    Blank lines separate paragraphs. A line that starts a list item is its own
    unit. Plain prose is split into sentences.
    """
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        current: list[str] = []
        for raw in paragraph.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue  # headings are context, not answers
            if TABLE_ROW_RE.match(line):
                units.extend(_sentences(" ".join(current)))
                units.append(line)  # a linearised table row is one unit, never split
                current = []
            elif LIST_ITEM_RE.match(line):
                units.extend(_sentences(" ".join(current)))
                current = [LIST_ITEM_RE.sub("", line)]
            else:
                current.append(line)  # prose, or the continuation of a list item
        units.extend(_sentences(" ".join(current)))
    return [u for u in units if len(tokens(u)) >= 2]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def detect_language(text: str) -> str:
    """Return 'es' or 'en' from function words and Spanish-only characters."""
    raw = text.lower()
    words = tokens(text)
    es = sum(w in _ES_MARKERS for w in words) + 2 * sum(ch in "¿¡ñ" for ch in raw)
    es += sum(ch in "áéíóú" for ch in raw)
    en = sum(w in _EN_MARKERS for w in words)
    return "es" if es > en else "en"


def jaccard(a: set[str], b: set[str]) -> float:
    """Set overlap in [0, 1]."""
    return len(a & b) / len(a | b) if a | b else 0.0
