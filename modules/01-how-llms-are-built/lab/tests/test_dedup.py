from __future__ import annotations

import random

from m01_foundations.dedup import (
    MinHasher,
    UnionFind,
    content_hash,
    estimated_jaccard,
    jaccard,
    lsh_candidates,
    lsh_threshold,
    near_duplicate_groups,
    shingles,
)

WORDS = ("plug bulb sensor hub camera thermostat room kitchen schedule night morning light energy "
         "power standby network channel router battery door window scene timer").split()


def _text(seed: int, n: int = 120) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n))


def test_content_hash_ignores_case_and_spacing() -> None:
    assert content_hash("Hello   World\n") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("hello world!")


def test_shingles_and_jaccard() -> None:
    assert shingles("a b c d e f", n=5) == {"a b c d e", "b c d e f"}
    assert shingles("a b", n=5) == {"a b"}
    assert jaccard({"x", "y"}, {"y", "z"}) == 1 / 3


def test_minhash_estimate_is_close_to_exact_jaccard() -> None:
    base = _text(1)
    edited = base + " updated with reader feedback"
    sa, sb = shingles(base), shingles(edited)
    hasher = MinHasher(num_perm=256)
    est = estimated_jaccard(hasher.signature(sa), hasher.signature(sb))
    assert abs(est - jaccard(sa, sb)) < 0.06
    unrelated = shingles(_text(2))
    assert estimated_jaccard(hasher.signature(sa), hasher.signature(unrelated)) < 0.1


def test_minhash_is_deterministic() -> None:
    s = shingles(_text(3))
    assert (MinHasher(seed=7).signature(s) == MinHasher(seed=7).signature(s)).all()


def test_lsh_finds_near_duplicates_only() -> None:
    a = _text(10)
    texts = [a, a + " small edit at the end", _text(11), _text(12)]
    hasher = MinHasher()
    sigs = [hasher.signature(shingles(t)) for t in texts]
    assert (0, 1) in lsh_candidates(sigs)
    assert near_duplicate_groups(texts) == [[0, 1]]


def test_lsh_threshold_formula() -> None:
    assert round(lsh_threshold(16, 8), 3) == 0.707


def test_union_find_groups() -> None:
    uf = UnionFind(5)
    uf.union(0, 3)
    uf.union(3, 4)
    assert sorted(uf.groups()) == [[0, 3, 4], [1], [2]]
