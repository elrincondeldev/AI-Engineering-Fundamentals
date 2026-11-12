from __future__ import annotations

import math

import numpy as np

from m01_foundations.transformer import (
    TinyConfig,
    causal_mask,
    cross_entropy,
    forward,
    init_params,
    layer_norm,
    rope,
    scaled_dot_product_attention,
    softmax,
)

CFG = TinyConfig(vocab_size=50, d_model=16, n_heads=4, n_layers=2, d_ff=32, context_length=12)


def test_softmax_rows_sum_to_one_and_is_stable() -> None:
    p = softmax(np.array([[1000.0, 1001.0], [0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[1], [0.5, 0.5])


def test_causal_mask() -> None:
    assert causal_mask(3).tolist() == [[True, False, False], [True, True, False], [True, True, True]]


def test_layer_norm_gives_zero_mean_unit_variance() -> None:
    x = np.random.default_rng(0).normal(3, 5, (4, 16))
    y = layer_norm(x, np.ones(16), np.zeros(16))
    assert np.allclose(y.mean(axis=-1), 0, atol=1e-7)
    assert np.allclose(y.std(axis=-1), 1, atol=1e-3)


def test_attention_never_looks_at_the_future() -> None:
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(5, 8)) for _ in range(3))
    out1, w = scaled_dot_product_attention(q, k, v, causal_mask(5))
    assert np.allclose(np.triu(w, 1), 0)
    k2, v2 = k.copy(), v.copy()
    k2[4], v2[4] = 99.0, -99.0  # change only the last token
    out2, _ = scaled_dot_product_attention(q, k2, v2, causal_mask(5))
    assert np.allclose(out1[:4], out2[:4])


def test_attention_without_positions_ignores_order() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 8))
    perm = [2, 0, 3, 1]
    out, _ = scaled_dot_product_attention(x, x, x)
    out_perm, _ = scaled_dot_product_attention(x[perm], x[perm], x[perm])
    assert np.allclose(out[perm], out_perm)  # same outputs, just shuffled: no notion of order


def test_rope_scores_depend_only_on_relative_position() -> None:
    rng = np.random.default_rng(3)
    q, k = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))
    score = lambda m, n: float(rope(q, np.array([m]))[0] @ rope(k, np.array([n]))[0])  # noqa: E731
    assert math.isclose(score(5, 2), score(13, 10), rel_tol=1e-9)
    assert not math.isclose(score(5, 2), score(5, 4), rel_tol=1e-3)
    assert np.allclose(np.linalg.norm(rope(q, np.array([7.0]))), np.linalg.norm(q))  # rotation keeps length


def test_forward_shapes_and_initial_loss() -> None:
    p = init_params(CFG, seed=0)
    ids = np.array([1, 5, 7, 9, 2])
    logits, weights = forward(ids, p, CFG)
    assert logits.shape == (5, CFG.vocab_size)
    assert len(weights) == CFG.n_layers and weights[0].shape == (CFG.n_heads, 5, 5)
    assert abs(cross_entropy(logits[:-1], ids[1:]) - math.log(CFG.vocab_size)) < 0.1


def test_forward_is_causal_end_to_end() -> None:
    p = init_params(CFG, seed=0)
    a, _ = forward(np.array([1, 2, 3, 4]), p, CFG)
    b, _ = forward(np.array([1, 2, 3, 40]), p, CFG)
    assert np.allclose(a[:3], b[:3])
    assert not np.allclose(a[3], b[3])


def test_cross_entropy_of_uniform_logits_is_log_vocab() -> None:
    assert math.isclose(cross_entropy(np.zeros((3, 10)), np.array([0, 4, 9])), math.log(10))


def test_context_length_is_enforced() -> None:
    p = init_params(CFG)
    try:
        forward(np.zeros(CFG.context_length + 1, dtype=int), p, CFG)
    except ValueError:
        return
    raise AssertionError("expected a ValueError")
