"""LoRA from scratch in numpy, on a linear intent classifier.

The base model is a softmax classifier ``logits = X @ W.T + b`` over hashed bag-of-words
features (D buckets, C = 8 intents). It is trained on pre-launch tickets and then frozen.
To adapt it to post-launch tickets we compare:

- full fine-tuning: update every entry of W and b (C*D + C numbers);
- LoRA: keep W frozen and learn ``W' = W + (alpha / r) * B @ A`` with
  B of shape (C, r) and A of shape (r, D), that is r * (C + D) numbers.

Everything is deterministic: features use a stable hash and A is initialised from
``numpy.random.default_rng(seed)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .textfeat import stable_hash, stems
from .tickets import INTENTS, Ticket, clean_text

DIM = 1024
LABEL_INDEX = {label: i for i, label in enumerate(INTENTS)}


def featurize(tickets: list[Ticket], dim: int = DIM) -> np.ndarray:
    """Binary hashed bag of word stems: unseen words still land in some bucket."""
    X = np.zeros((len(tickets), dim))
    for row, t in enumerate(tickets):
        for s in stems(clean_text(t.body)):
            X[row, stable_hash(s) % dim] = 1.0
    return X


def labels_of(tickets: list[Ticket]) -> np.ndarray:
    return np.array([LABEL_INDEX[t.intent] for t in tickets])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def error_signal(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of the mean cross-entropy with respect to the logits: (p - one_hot(y)) / n."""
    g = softmax(logits)
    g[np.arange(len(y)), y] -= 1.0
    return g / len(y)


@dataclass
class LinearClassifier:
    W: np.ndarray  # (C, D)
    b: np.ndarray  # (C,)

    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W.T + self.b

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return round(float((self.logits(X).argmax(axis=1) == y).mean()), 3)

    @property
    def n_params(self) -> int:
        return self.W.size + self.b.size


def train_full(
    X: np.ndarray, y: np.ndarray, start: LinearClassifier | None = None,
    steps: int = 500, lr: float = 1.0, l2: float = 1e-4,
) -> LinearClassifier:
    """Gradient descent on every weight. From scratch (``start=None``) or from a base model."""
    C, D = len(INTENTS), X.shape[1]
    W = np.zeros((C, D)) if start is None else start.W.copy()
    b = np.zeros(C) if start is None else start.b.copy()
    for _ in range(steps):
        G = error_signal(X @ W.T + b, y)
        W -= lr * (G.T @ X + l2 * W)
        b -= lr * G.sum(axis=0)
    return LinearClassifier(W, b)


@dataclass
class LoRAClassifier:
    """A frozen base plus a trainable low-rank update."""

    base: LinearClassifier
    A: np.ndarray  # (r, D) random at start
    B: np.ndarray  # (C, r) zeros at start, so the adapted model starts identical to the base
    alpha: float

    @property
    def rank(self) -> int:
        return self.A.shape[0]

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def delta(self) -> np.ndarray:
        return self.scale * self.B @ self.A

    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.base.W + self.delta()).T + self.base.b

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return round(float((self.logits(X).argmax(axis=1) == y).mean()), 3)

    @property
    def n_params(self) -> int:
        return self.A.size + self.B.size

    def merged(self) -> LinearClassifier:
        """Fold the update into W: same predictions, zero extra cost at inference time."""
        return LinearClassifier(self.base.W + self.delta(), self.base.b.copy())


def train_lora(
    base: LinearClassifier, X: np.ndarray, y: np.ndarray, rank: int,
    alpha: float | None = None, steps: int = 100, lr: float = 2.0, seed: int = 0,
) -> LoRAClassifier:
    """Gradient descent on A and B only; the base weights never change.

    ``alpha`` defaults to ``rank`` so that the scale alpha / r is 1 at every rank.
    """
    rng = np.random.default_rng(seed)
    C, D = base.W.shape
    A = rng.normal(0.0, 1.0 / np.sqrt(D), size=(rank, D))
    model = LoRAClassifier(base, A, np.zeros((C, rank)), float(rank if alpha is None else alpha))
    for _ in range(steps):
        G = error_signal(model.logits(X), y)
        grad_W = G.T @ X  # (C, D): what full fine-tuning would apply to W
        grad_B = model.scale * grad_W @ model.A.T  # chain rule through W' = W + s * B @ A
        grad_A = model.scale * model.B.T @ grad_W
        model.B -= lr * grad_B
        model.A -= lr * grad_A
    return model


def lora_params(d_in: int, d_out: int, rank: int) -> int:
    """Trainable numbers in one LoRA pair: B is (d_out, r) and A is (r, d_in)."""
    return rank * (d_in + d_out)


def full_params(d_in: int, d_out: int) -> int:
    """Trainable numbers when the whole weight matrix is updated (no bias)."""
    return d_in * d_out


def weight_memory_gb(n_params: float, bits: int) -> float:
    """Memory to hold the weights alone, in gigabytes (10**9 bytes)."""
    return round(n_params * bits / 8 / 1e9, 2)
