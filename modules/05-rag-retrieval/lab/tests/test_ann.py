"""Exact k-NN, recall, and the numpy IVF."""

from __future__ import annotations

import numpy as np

from m05_retrieval.ann import NumpyIVF, exact_knn, make_vectors, recall_at_k


def test_vectors_are_seeded_and_unit_length() -> None:
    a, qa = make_vectors(n=500, dim=16, n_queries=5)
    b, qb = make_vectors(n=500, dim=16, n_queries=5)
    assert np.array_equal(a, b) and np.array_equal(qa, qb)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_exact_knn_matches_full_sort() -> None:
    base, queries = make_vectors(n=300, dim=8, n_queries=4)
    ids = exact_knn(base, queries, k=5)
    full = np.argsort(-(queries @ base.T), axis=1)[:, :5]
    assert np.array_equal(ids, full)


def test_recall_counts_overlap() -> None:
    truth = np.array([[1, 2, 3, 4]])
    assert recall_at_k(np.array([[4, 3, 9, 8]]), truth) == 0.5
    assert recall_at_k(np.array([[-1, 1, 2, 3]]), truth) == 0.75


def test_ivf_probing_every_list_is_exact() -> None:
    base, queries = make_vectors(n=2000, dim=16, n_queries=20)
    ivf = NumpyIVF(base, n_lists=16)
    truth = exact_knn(base, queries, 10)
    ids_all, compared = ivf.search(queries, 10, nprobe=16)
    assert recall_at_k(ids_all, truth) == 1.0 and compared == 2000 + 16
    ids_one, compared_one = ivf.search(queries, 10, nprobe=1)
    assert recall_at_k(ids_one, truth) < 1.0 and compared_one < 2000
