from __future__ import annotations

import math

import numpy as np

from m01_foundations.neural import NeuralBigram, count_bigram_loss, train_neural_bigram


def test_gradient_matches_finite_differences() -> None:
    """The hand-written gradient must agree with a numerical estimate: the classic backprop check."""
    x, y = np.array([0, 1, 2, 1]), np.array([1, 2, 0, 0])
    model = NeuralBigram(vocab_size=3, dim=4, seed=0)
    w = model.out.copy()
    eps = 1e-6
    model.out[1, 2] = w[1, 2] + eps
    plus = model.loss(x, y)
    model.out[1, 2] = w[1, 2] - eps
    minus = model.loss(x, y)
    model.out[:] = w
    numeric = (plus - minus) / (2 * eps)
    before = model.out[1, 2]
    model.step(x, y, lr=1.0)  # with lr = 1 the update equals minus the gradient
    analytic = before - model.out[1, 2]
    assert math.isclose(numeric, analytic, rel_tol=1e-4, abs_tol=1e-9)


def test_training_lowers_the_loss_deterministically() -> None:
    rng = np.random.default_rng(0)
    pattern = np.tile(np.arange(8), 200)  # 0 1 2 ... 7 0 1 2 ...: perfectly predictable
    noise = rng.integers(0, 8, size=50)
    data = np.concatenate([pattern, noise])
    a = train_neural_bigram(data, pattern[:200], 8, dim=8, steps=200, batch=64, lr=2.0, eval_every=50)
    b = train_neural_bigram(data, pattern[:200], 8, dim=8, steps=200, batch=64, lr=2.0, eval_every=50)
    assert a.history == b.history
    assert a.history[0][2] > 2.0 > 0.5 > a.history[-1][2]
    assert math.isclose(a.uniform_loss, math.log(8))


def test_count_bigram_reference() -> None:
    seq = np.array([0, 1, 0, 1, 0, 1])
    assert count_bigram_loss(seq, seq, vocab_size=2, alpha=1e-9) < 1e-6
