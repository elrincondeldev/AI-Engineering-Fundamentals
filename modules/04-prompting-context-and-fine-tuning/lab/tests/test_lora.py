"""LoRA maths: initial state, gradients, merging and parameter counts."""

from __future__ import annotations

import numpy as np

from m04_adaptation.lora import (
    LinearClassifier,
    LoRAClassifier,
    error_signal,
    full_params,
    lora_params,
    train_full,
    train_lora,
    weight_memory_gb,
)


def _toy(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = (rng.random((40, 30)) < 0.2).astype(float)
    y = rng.integers(0, 8, size=40)
    return X, y


def _loss(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def test_lora_starts_identical_to_the_base_and_leaves_it_frozen() -> None:
    X, y = _toy()
    base = train_full(X, y, steps=50)
    W_before = base.W.copy()
    model = train_lora(base, X, y, rank=2, steps=0)
    assert np.allclose(model.logits(X), base.logits(X))  # B starts at zero
    trained = train_lora(base, X, y, rank=2, steps=20)
    assert np.array_equal(base.W, W_before)
    assert not np.allclose(trained.logits(X), base.logits(X))


def test_analytic_gradients_match_finite_differences() -> None:
    X, y = _toy(1)
    rng = np.random.default_rng(2)
    base = LinearClassifier(rng.normal(size=(8, 30)), rng.normal(size=8))
    m = LoRAClassifier(base, rng.normal(size=(3, 30)), rng.normal(size=(8, 3)), alpha=6.0)
    grad_W = error_signal(m.logits(X), y).T @ X
    grad_B = m.scale * grad_W @ m.A.T
    grad_A = m.scale * m.B.T @ grad_W
    eps = 1e-6
    for name, matrix, grad in (("A", m.A, grad_A), ("B", m.B, grad_B)):
        i, j = 1, 2
        old = matrix[i, j]
        matrix[i, j] = old + eps
        up = _loss(m.logits(X), y)
        matrix[i, j] = old - eps
        down = _loss(m.logits(X), y)
        matrix[i, j] = old
        assert abs((up - down) / (2 * eps) - grad[i, j]) < 1e-6, name


def test_merging_gives_identical_predictions() -> None:
    X, y = _toy()
    base = train_full(X, y, steps=50)
    m = train_lora(base, X, y, rank=4, steps=30)
    assert np.allclose(m.merged().logits(X), m.logits(X))


def test_parameter_counts() -> None:
    assert lora_params(4096, 4096, 8) == 65_536
    assert full_params(4096, 4096) == 16_777_216
    assert lora_params(1024, 8, 1) == 1032
    X, y = _toy()
    assert train_lora(train_full(X, y, steps=1), X, y, rank=3, steps=1).n_params == 3 * (30 + 8)
    assert weight_memory_gb(7e9, 4) == 3.5
