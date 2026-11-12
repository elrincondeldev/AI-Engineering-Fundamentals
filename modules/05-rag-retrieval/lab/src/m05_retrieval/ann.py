"""Exact vs approximate nearest-neighbour search on a seeded synthetic vector set.

The help centre has about 200 chunks, far too few for approximate search to
matter, so this module generates a larger, clustered set of unit vectors (like
real embeddings, they are not uniformly spread) and measures recall@10 against
exact search, memory and the number of vectors each method compares per query.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

try:  # FAISS is a declared dependency, but the numpy parts work without it.
    import faiss
except ImportError:  # pragma: no cover - exercised only when the wheel is missing
    faiss = None


@dataclass(frozen=True)
class AnnResult:
    method: str
    params: str
    recall_at_10: float
    compared_per_query: int  # similarities computed per query (centroids and graph hops included)
    memory_bytes: int
    ms_per_query: float  # wall-clock, varies by machine: never asserted


def make_vectors(
    n: int = 50_000,
    dim: int = 64,
    n_queries: int = 200,
    n_topics: int = 200,
    spread: float = 1.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors clustered around ``n_topics`` random centres; queries come from the same mixture."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_topics, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    def sample(count: int) -> np.ndarray:
        topic = rng.integers(0, n_topics, size=count)
        pts = centres[topic] + spread * rng.standard_normal((count, dim)) / np.sqrt(dim)
        return (pts / np.linalg.norm(pts, axis=1, keepdims=True)).astype(np.float32)

    return sample(n), sample(n_queries)


def exact_knn(base: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    """Brute force: score every vector with a dot product, keep the k best (cosine on unit vectors)."""
    scores = queries @ base.T  # (n_queries, n)
    top = np.argpartition(-scores, k, axis=1)[:, :k]  # k best, unordered
    rows = np.arange(len(queries))[:, None]
    order = np.argsort(-scores[rows, top], axis=1, kind="stable")
    return top[rows, order]


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    """Average share of the true k nearest neighbours that the method returned."""
    k = truth.shape[1]
    hits = sum(len(set(f[f >= 0].tolist()) & set(t.tolist())) for f, t in zip(found, truth))
    return round(hits / (k * len(truth)), 3)


# --------------------------------------------------------------------------- #
# A small IVF in numpy (the idea behind faiss.IndexIVFFlat)                   #
# --------------------------------------------------------------------------- #


def kmeans(x: np.ndarray, n_clusters: int, iters: int = 10, seed: int = 0) -> np.ndarray:
    """Plain spherical k-means: returns unit-length centroids."""
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(len(x), n_clusters, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(x @ cent.T, axis=1)
        for c in range(n_clusters):
            members = x[assign == c]
            if len(members):
                cent[c] = members.sum(axis=0)
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    return cent


class NumpyIVF:
    """Inverted file index: cluster the vectors, then search only the ``nprobe`` closest clusters."""

    def __init__(self, base: np.ndarray, n_lists: int = 256, seed: int = 0) -> None:
        self.base = base
        self.centroids = kmeans(base, n_lists, seed=seed)
        assign = np.argmax(base @ self.centroids.T, axis=1)
        self.lists = [np.flatnonzero(assign == c) for c in range(n_lists)]

    def search(self, queries: np.ndarray, k: int = 10, nprobe: int = 8) -> tuple[np.ndarray, float]:
        """Return (ids, average number of similarities computed per query, centroids included)."""
        out = np.full((len(queries), k), -1, dtype=np.int64)
        compared = len(self.centroids) * len(queries)
        probes = np.argsort(-(queries @ self.centroids.T), axis=1)[:, :nprobe]
        for qi, q in enumerate(queries):
            cand = np.concatenate([self.lists[c] for c in probes[qi]])
            compared += len(cand)
            scores = self.base[cand] @ q
            best = cand[np.argsort(-scores, kind="stable")[:k]]
            out[qi, : len(best)] = best
        return out, compared / len(queries)


# --------------------------------------------------------------------------- #
# The benchmark                                                               #
# --------------------------------------------------------------------------- #


def _timed(fn, n_queries: int):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start) * 1000 / n_queries


def run_benchmark(n: int = 50_000, dim: int = 64, n_queries: int = 200, k: int = 10, seed: int = 0,
                  use_faiss: bool = True, spread: float = 1.5) -> list[AnnResult]:
    """Exact numpy search, numpy IVF and the FAISS indexes, all scored against exact search."""
    base, queries = make_vectors(n, dim, n_queries, spread=spread, seed=seed)
    truth, ms = _timed(lambda: exact_knn(base, queries, k), n_queries)
    results = [AnnResult("numpy exact", "-", 1.0, n, base.nbytes, ms)]
    ivf = NumpyIVF(base, n_lists=256, seed=seed)
    for nprobe in (1, 8):
        (ids, compared), ms = _timed(lambda p=nprobe: ivf.search(queries, k, p), n_queries)
        mem = base.nbytes + ivf.centroids.nbytes + 8 * n
        results.append(AnnResult("numpy IVF", f"nlist=256 nprobe={nprobe}", recall_at_k(ids, truth),
                                 round(compared), mem, ms))
    if use_faiss and faiss is not None:
        results.extend(_faiss_suite(base, queries, truth, k))
    return results


def _faiss_bytes(index) -> int:  # type: ignore[no-untyped-def]
    return int(faiss.serialize_index(index).nbytes)


def _faiss_suite(base: np.ndarray, queries: np.ndarray, truth: np.ndarray, k: int) -> list[AnnResult]:
    faiss.omp_set_num_threads(1)  # one thread: identical graphs and results on every run
    n, dim = base.shape
    nq = len(queries)
    out: list[AnnResult] = []

    flat = faiss.IndexFlatIP(dim)
    flat.add(base)
    (_, ids), ms = _timed(lambda: flat.search(queries, k), nq)
    out.append(AnnResult("faiss IndexFlatIP", "-", recall_at_k(ids, truth), n, _faiss_bytes(flat), ms))

    quant = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quant, dim, 256, faiss.METRIC_INNER_PRODUCT)
    ivf.cp.seed = 0
    ivf.train(base)
    ivf.add(base)
    sizes = np.array([ivf.invlists.list_size(i) for i in range(ivf.nlist)])
    for nprobe in (1, 8, 32):
        ivf.nprobe = nprobe
        (_, ids), ms = _timed(lambda: ivf.search(queries, k), nq)
        _, probes = quant.search(queries, nprobe)
        compared = int(round(sizes[probes].sum() / nq)) + ivf.nlist
        out.append(AnnResult("faiss IndexIVFFlat", f"nlist=256 nprobe={nprobe}", recall_at_k(ids, truth),
                             compared, _faiss_bytes(ivf), ms))

    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 64
    hnsw.add(base)
    for ef in (16, 64, 128):
        hnsw.hnsw.efSearch = ef
        faiss.cvar.hnsw_stats.reset()
        (_, ids), ms = _timed(lambda: hnsw.search(queries, k), nq)
        compared = int(round(faiss.cvar.hnsw_stats.ndis / nq))
        out.append(AnnResult("faiss IndexHNSWFlat", f"M=32 efSearch={ef}", recall_at_k(ids, truth),
                             compared, _faiss_bytes(hnsw), ms))

    quant_pq = faiss.IndexFlatIP(dim)
    ivfpq = faiss.IndexIVFPQ(quant_pq, dim, 256, 16, 8, faiss.METRIC_INNER_PRODUCT)
    ivfpq.cp.seed = 0
    ivfpq.train(base)
    ivfpq.add(base)
    ivfpq.nprobe = 8
    (_, ids), ms = _timed(lambda: ivfpq.search(queries, k), nq)
    _, probes = quant_pq.search(queries, 8)
    pq_sizes = np.array([ivfpq.invlists.list_size(i) for i in range(ivfpq.nlist)])
    out.append(AnnResult("faiss IndexIVFPQ", "nlist=256 m=16 nprobe=8", recall_at_k(ids, truth),
                         int(round(pq_sizes[probes].sum() / nq)) + ivfpq.nlist, _faiss_bytes(ivfpq), ms))
    return out
