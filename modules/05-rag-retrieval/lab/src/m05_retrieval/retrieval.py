"""Retrievers behind one interface, and a config that builds the whole offline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .bm25 import BM25
from .chunking import Chunk, CorpusReport, chunk_corpus, prepare_corpus
from .hybrid import rrf_fuse
from .parsing import Document, load_kb
from .rerank import rerank
from .vectors import Embedder, HashingEmbedder, VectorIndex


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    chunks: list[Chunk]

    def search(self, query: str, k: int = 5) -> list[Hit]: ...


def _top(scores: np.ndarray | list[float], k: int) -> list[int]:
    """Indices of the k best scores (stable: earlier chunks win ties). Zero scores are dropped."""
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-arr, kind="stable")[:k]
    return [int(i) for i in order if arr[i] > 0]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], *, fold: bool = False) -> None:
        self.chunks = chunks
        self.bm25 = BM25([c.text for c in chunks], fold=fold)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        s = self.bm25.scores(query)
        return [Hit(self.chunks[i], s[i]) for i in _top(s, k)]


class VectorRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None) -> None:
        self.chunks = chunks
        self.index = VectorIndex([c.text for c in chunks], embedder or HashingEmbedder())

    def search(self, query: str, k: int = 5) -> list[Hit]:
        s = self.index.scores(query)
        return [Hit(self.chunks[i], float(s[i])) for i in _top(s, k)]


class HybridRetriever:
    """Run several retrievers, fuse their rankings with RRF."""

    def __init__(self, retrievers: list[Retriever], *, depth: int = 50, k_rrf: int = 60) -> None:
        self.retrievers = retrievers
        self.chunks = retrievers[0].chunks
        self.depth, self.k_rrf = depth, k_rrf
        self._pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        rankings = [
            [self._pos[h.chunk.chunk_id] for h in r.search(query, self.depth)] for r in self.retrievers
        ]
        return [Hit(self.chunks[i], s) for i, s in rrf_fuse(rankings, self.k_rrf)[:k]]


class RerankedRetriever:
    """Take ``candidates`` hits from a first-stage retriever and reorder them with the lexical reranker."""

    def __init__(self, base: Retriever, *, candidates: int = 20) -> None:
        self.base, self.candidates, self.chunks = base, candidates, base.chunks

    def search(self, query: str, k: int = 5) -> list[Hit]:
        first = self.base.search(query, self.candidates)
        order = rerank(query, [(i, h.chunk.text, h.score) for i, h in enumerate(first)])
        return [Hit(first[i].chunk, s) for i, s in order[:k]]


# --------------------------------------------------------------------------- #
# Pipeline configuration                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    """Everything that changes retrieval quality, in one place."""

    retriever: str = "bm25"  # bm25 | vector | hybrid
    chunker: str = "heading"  # fixed | recursive | heading | table | semantic
    filter_meta: bool = True  # drop superseded and internal documents
    dedupe: bool = True  # drop near-duplicate documents
    title: bool = False  # prefix chunks with the document title
    fold: bool = False  # fold accents in BM25 (México == Mexico)
    rerank: bool = False  # lexical reranking of the top 20

    @property
    def name(self) -> str:
        parts = [self.retriever, self.chunker]
        if not self.filter_meta:
            parts.append("nofilter")
        if self.title:
            parts.append("title")
        if self.fold:
            parts.append("fold")
        if self.rerank:
            parts.append("rerank")
        return "/".join(parts)


@dataclass
class Pipeline:
    config: Config
    docs: list[Document]
    report: CorpusReport
    chunks: list[Chunk]
    retriever: Retriever


def build_pipeline(config: Config = Config(), docs: list[Document] | None = None,
                   embedder: Embedder | None = None) -> Pipeline:
    """Parse -> filter -> chunk -> index, following ``config``."""
    all_docs = docs if docs is not None else load_kb()
    kept, report = prepare_corpus(all_docs, filter_meta=config.filter_meta, dedupe=config.dedupe)
    chunks = chunk_corpus(kept, config.chunker, title=config.title)
    retriever: Retriever
    if config.retriever == "bm25":
        retriever = BM25Retriever(chunks, fold=config.fold)
    elif config.retriever == "vector":
        retriever = VectorRetriever(chunks, embedder)
    elif config.retriever == "hybrid":
        retriever = HybridRetriever(
            [BM25Retriever(chunks, fold=config.fold), VectorRetriever(chunks, embedder)]
        )
    else:
        raise ValueError(f"unknown retriever {config.retriever!r}")
    if config.rerank:
        retriever = RerankedRetriever(retriever)
    return Pipeline(config, all_docs, report, chunks, retriever)
