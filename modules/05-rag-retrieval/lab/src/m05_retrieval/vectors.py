"""Dense retrieval: embed chunks once, embed the query, rank by cosine similarity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .providers import get_provider, hashing_embedding


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class HashingEmbedder:
    """Offline bag-of-words + character-trigram embedder from ``providers.py`` (no model, no network)."""

    dim: int = 256
    name: str = "hashing-256"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([hashing_embedding(t, self.dim) for t in texts], dtype=np.float32)


@dataclass
class ProviderEmbedder:
    """Embeddings from the configured provider (``LLM_PROVIDER=ollama`` or ``openai``). Not used in tests."""

    provider_name: str | None = None
    name: str = "provider"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        provider = get_provider(self.provider_name)
        vecs = np.array(provider.embed(list(texts)), dtype=np.float32)
        return normalize(vecs)


def normalize(vecs: np.ndarray) -> np.ndarray:
    """Scale every row to length 1 so that a dot product equals cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


class VectorIndex:
    """A flat (exact) vector index: one row per chunk, searched by matrix product."""

    def __init__(self, texts: Sequence[str], embedder: Embedder) -> None:
        self.embedder = embedder
        self.matrix = normalize(embedder.embed(texts))

    @property
    def nbytes(self) -> int:
        return int(self.matrix.nbytes)

    def scores(self, query: str) -> np.ndarray:
        q = normalize(self.embedder.embed([query]))[0]
        return self.matrix @ q

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in top]
