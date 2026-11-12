"""Tiny, deterministic text features shared by the toy learner, the selector and LoRA."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
STOPWORDS = frozenset(
    """a an and are at be but by can do does for from has have hi hello hey how i if in is it its me my
    no not of on or our please so that the this to was what when where why will with you your thanks thank
    cheers urgent good morning hola buenas gracias el la los las de del en un una que mi me por para con es
    se lo al y o ya no""".split()
)


def fold(text: str) -> str:
    """Lower-case and remove accents so that 'conexión' and 'conexion' match."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def stems(text: str, prefix: int = 5) -> set[str]:
    """Content words cut to their first ``prefix`` letters: a crude, language-agnostic stemmer."""
    words = _WORD_RE.findall(fold(text))
    return {w[:prefix] for w in words if len(w) > 1 and w not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    """Shared items divided by all distinct items (0 = nothing shared, 1 = identical sets)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def stable_hash(text: str) -> int:
    """A hash that is identical on every machine and every run (unlike Python's ``hash``)."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")
