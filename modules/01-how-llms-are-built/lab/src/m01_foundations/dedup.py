"""Exact and near-duplicate detection: content hashing, MinHash and LSH banding.

Exact duplicates are found by hashing normalised text. Near duplicates (the same article with a
few edits) are found with MinHash: each document becomes a short signature whose positions agree
with probability equal to the Jaccard similarity of the documents' shingle sets. LSH banding then
proposes candidate pairs without comparing every pair of documents.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PRIME = (1 << 31) - 1  # a Mersenne prime; keeps a * x + b inside 64-bit integers


def normalise_text(text: str) -> str:
    """Lowercase and collapse whitespace so trivial formatting changes do not hide a duplicate."""
    return " ".join(text.lower().split())


def content_hash(text: str) -> str:
    """SHA-256 of the normalised text: equal hashes mean exact duplicates."""
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


def shingles(text: str, n: int = 5) -> set[str]:
    """Word n-grams ("shingles") of a text, lowercased. Short texts give one shingle."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Exact Jaccard similarity |A ∩ B| / |A ∪ B| (1.0 for two empty sets)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _shingle_ids(items: set[str]) -> np.ndarray:
    """Map each shingle to a stable 31-bit integer (Python's hash() is salted per process)."""
    ids = [int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little") % _PRIME
           for s in sorted(items)]
    return np.array(ids, dtype=np.int64)


@dataclass(frozen=True)
class MinHasher:
    """MinHash with ``num_perm`` random linear hash functions h(x) = (a * x + b) mod p."""

    num_perm: int = 128
    seed: int = 1

    def _params(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        a = rng.integers(1, _PRIME, size=self.num_perm, dtype=np.int64)
        b = rng.integers(0, _PRIME, size=self.num_perm, dtype=np.int64)
        return a, b

    def signature(self, items: set[str]) -> np.ndarray:
        """Return the MinHash signature: for every hash function, the minimum over the shingles."""
        if not items:
            return np.full(self.num_perm, _PRIME, dtype=np.int64)
        a, b = self._params()
        x = _shingle_ids(items)
        hashed = (a[:, None] * x[None, :] + b[:, None]) % _PRIME  # (num_perm, n_shingles)
        return hashed.min(axis=1)


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Share of signature positions that agree: an unbiased estimate of the Jaccard similarity."""
    return float(np.mean(sig_a == sig_b))


def lsh_candidates(signatures: Sequence[np.ndarray], bands: int = 16) -> set[tuple[int, int]]:
    """Split each signature into ``bands`` bands; documents sharing any whole band become candidates."""
    if not signatures:
        return set()
    rows = len(signatures[0]) // bands
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for idx, sig in enumerate(signatures):
            key = sig[band * rows : (band + 1) * rows].tobytes()
            buckets[key].append(idx)
        for members in buckets.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.add((members[i], members[j]))
    return pairs


def lsh_threshold(bands: int, rows: int) -> float:
    """Approximate Jaccard similarity at which a pair has a 50% chance of becoming a candidate."""
    return (1 / bands) ** (1 / rows)


class UnionFind:
    """Tiny disjoint-set structure to group documents into duplicate clusters."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return [sorted(g) for g in out.values()]


def near_duplicate_groups(
    texts: Sequence[str],
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    bands: int = 16,
    shingle_size: int = 5,
    seed: int = 1,
) -> list[list[int]]:
    """Group texts whose estimated Jaccard similarity is at least ``threshold``.

    Returns only groups with two or more members, as lists of indices into ``texts``.
    """
    hasher = MinHasher(num_perm=num_perm, seed=seed)
    sigs = [hasher.signature(shingles(t, shingle_size)) for t in texts]
    uf = UnionFind(len(texts))
    for i, j in sorted(lsh_candidates(sigs, bands=bands)):
        if estimated_jaccard(sigs[i], sigs[j]) >= threshold:
            uf.union(i, j)
    return [g for g in uf.groups() if len(g) > 1]
