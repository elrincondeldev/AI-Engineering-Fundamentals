"""A byte-level byte-pair-encoding (BPE) tokenizer, trained from scratch.

Every text is first turned into UTF-8 bytes, so any string can be encoded (no "unknown" token).
Training repeatedly finds the most frequent adjacent pair of token ids and merges it into a new id.
Encoding replays those merges in the order they were learned; decoding concatenates bytes.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

# Pre-tokenisation: split text into word-like chunks so merges never cross a word boundary.
# A leading space stays attached to the word, as in GPT-style tokenizers.
SPLIT_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+""")
END_OF_TEXT = "<|endoftext|>"


def pretokenize(text: str) -> list[str]:
    return SPLIT_RE.findall(text)


def pair_counts(chunks: dict[tuple[int, ...], int]) -> Counter[tuple[int, int]]:
    """Count adjacent id pairs, weighting each chunk by how often it occurs."""
    counts: Counter[tuple[int, int]] = Counter()
    for ids, freq in chunks.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += freq
    return counts


def merge_pair(ids: Sequence[int], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids`` with ``new_id``, left to right."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


@dataclass
class BPETokenizer:
    """Learned merges plus the byte string behind every id."""

    merges: dict[tuple[int, int], int] = field(default_factory=dict)  # insertion order = merge rank
    special: dict[str, int] = field(default_factory=dict)
    vocab: dict[int, bytes] = field(default_factory=lambda: {i: bytes([i]) for i in range(256)})
    _cache: dict[str, tuple[int, ...]] = field(default_factory=dict, repr=False)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special)

    def truncated(self, n_merges: int) -> BPETokenizer:
        """The same tokenizer with only its first ``n_merges`` merges (a smaller vocabulary)."""
        kept = dict(list(self.merges.items())[:n_merges])
        vocab = {i: b for i, b in self.vocab.items() if i < 256 + n_merges}
        special = {s: 256 + n_merges + k for k, s in enumerate(self.special)}
        return BPETokenizer(merges=kept, special=special, vocab=vocab)

    def _encode_chunk(self, chunk: str) -> tuple[int, ...]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids: tuple[int, ...] = tuple(chunk.encode("utf-8"))
        while len(ids) >= 2:
            ranked = [(self.merges[p], p) for p in zip(ids, ids[1:]) if p in self.merges]
            if not ranked:
                break
            new_id, pair = min(ranked)  # the earliest-learned merge applies first
            ids = merge_pair(ids, pair, new_id)
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Text to token ids. Special tokens are matched first and never split."""
        if allow_special and self.special:
            pattern = "(" + "|".join(re.escape(s) for s in self.special) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
        out: list[int] = []
        for part in parts:
            if part in self.special and allow_special:
                out.append(self.special[part])
                continue
            for chunk in pretokenize(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        """Token ids back to text. Invalid UTF-8 (a split multi-byte character) becomes U+FFFD."""
        inverse = {i: s.encode("utf-8") for s, i in self.special.items()}
        data = b"".join(inverse[i] if i in inverse else self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def token_strings(self, ids: Iterable[int]) -> list[str]:
        """Human-readable pieces, one per id. Spaces show as '·'; a lone byte of a character as <C3>."""
        inverse = {i: s for s, i in self.special.items()}
        out = []
        for i in ids:
            if i in inverse:
                out.append(inverse[i])
                continue
            text = self.vocab[i].decode("utf-8", errors="backslashreplace")
            text = re.sub(r"\\x([0-9a-f]{2})", lambda m: f"<{m.group(1).upper()}>", text)
            out.append(text.replace(" ", "·"))
        return out


@dataclass(frozen=True)
class MergeStep:
    rank: int
    pair: tuple[int, int]
    new_id: int
    count: int
    piece: bytes


def train_bpe(
    texts: Iterable[str], vocab_size: int, special_tokens: Sequence[str] = (END_OF_TEXT,)
) -> tuple[BPETokenizer, list[MergeStep]]:
    """Learn ``vocab_size - 256 - len(special_tokens)`` merges from ``texts``."""
    n_merges = vocab_size - 256 - len(special_tokens)
    if n_merges < 0:
        raise ValueError("vocab_size must be at least 256 + the number of special tokens")
    chunk_freq = Counter(chunk for t in texts for chunk in pretokenize(t))
    chunks: dict[tuple[int, ...], int] = Counter()
    for chunk, freq in chunk_freq.items():
        chunks[tuple(chunk.encode("utf-8"))] += freq
    tok = BPETokenizer()
    history: list[MergeStep] = []
    for rank in range(n_merges):
        counts = pair_counts(chunks)
        if not counts:
            break
        pair, count = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))  # ties: smallest pair wins
        new_id = 256 + rank
        tok.merges[pair] = new_id
        tok.vocab[new_id] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        history.append(MergeStep(rank, pair, new_id, count, tok.vocab[new_id]))
        merged: dict[tuple[int, ...], int] = Counter()
        for ids, freq in chunks.items():
            merged[merge_pair(ids, pair, new_id) if len(ids) > 1 else ids] += freq
        chunks = merged
    tok.special = {s: 256 + len(tok.merges) + k for k, s in enumerate(special_tokens)}
    return tok, history


# --------------------------------------------------------------------------- #
# Measurements                                                                #
# --------------------------------------------------------------------------- #


def count_words(text: str) -> int:
    """Whitespace-separated words: a language-neutral unit for comparing tokenizers."""
    return len(text.split())


def tokens_per_word(tok: BPETokenizer, texts: Sequence[str]) -> float:
    tokens = sum(len(tok.encode(t, allow_special=False)) for t in texts)
    return tokens / sum(count_words(t) for t in texts)


def bytes_per_token(tok: BPETokenizer, texts: Sequence[str]) -> float:
    tokens = sum(len(tok.encode(t, allow_special=False)) for t in texts)
    return sum(len(t.encode("utf-8")) for t in texts) / tokens
