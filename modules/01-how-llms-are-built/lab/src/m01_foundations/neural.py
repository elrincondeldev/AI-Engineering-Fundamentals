"""Gradient descent on the smallest useful neural language model.

The model sees one token and predicts the next one: look up the token's embedding (a vector of
``dim`` numbers), multiply by an output matrix to get one score (logit) per vocabulary entry, and
apply softmax. Training nudges every weight a little in the direction that lowers the
cross-entropy loss. A transformer is trained the same way, only with many more layers in between.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .transformer import softmax


@dataclass
class NeuralBigram:
    vocab_size: int
    dim: int = 32
    seed: int = 0
    emb: np.ndarray = field(init=False)
    out: np.ndarray = field(init=False)
    bias: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.emb = rng.normal(0, 0.02, (self.vocab_size, self.dim))
        self.out = rng.normal(0, 0.02, (self.dim, self.vocab_size))
        self.bias = np.zeros(self.vocab_size)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.emb[x] @ self.out + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        probs = softmax(self.logits(x))
        return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())

    def step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        """One gradient-descent step on a batch. Returns the batch loss before the update."""
        h = self.emb[x]  # (B, dim)
        probs = softmax(h @ self.out + self.bias)  # (B, V)
        loss = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())
        grad_logits = probs
        grad_logits[np.arange(len(y)), y] -= 1.0  # d loss / d logits = softmax - one_hot
        grad_logits /= len(y)
        grad_out = h.T @ grad_logits
        grad_bias = grad_logits.sum(axis=0)
        grad_h = grad_logits @ self.out.T
        self.out -= lr * grad_out
        self.bias -= lr * grad_bias
        np.add.at(self.emb, x, -lr * grad_h)  # only the embeddings that were used get a gradient
        return loss


@dataclass(frozen=True)
class TrainResult:
    history: list[tuple[int, float, float]]  # (step, train batch loss, validation loss)
    uniform_loss: float
    count_bigram_loss: float


def count_bigram_loss(train: np.ndarray, valid: np.ndarray, vocab_size: int, alpha: float = 0.01) -> float:
    """Validation loss of a smoothed counting bigram model, a reference point for the network."""
    pair = Counter(zip(train[:-1].tolist(), train[1:].tolist()))
    first = Counter(train[:-1].tolist())
    total = 0.0
    for a, b in zip(valid[:-1].tolist(), valid[1:].tolist()):
        total -= math.log((pair[(a, b)] + alpha) / (first[a] + alpha * vocab_size))
    return total / (len(valid) - 1)


def train_neural_bigram(train: np.ndarray, valid: np.ndarray, vocab_size: int, *, dim: int = 32,
                        steps: int = 400, batch: int = 512, lr: float = 2.0, seed: int = 0,
                        eval_every: int = 100) -> TrainResult:
    """Mini-batch gradient descent with a fixed learning rate. Fully deterministic for a given seed."""
    model = NeuralBigram(vocab_size, dim, seed)
    rng = np.random.default_rng(seed)
    vx, vy = valid[:-1], valid[1:]
    history = [(0, model.loss(train[:-1][:batch], train[1:][:batch]), model.loss(vx, vy))]
    for s in range(1, steps + 1):
        idx = rng.integers(0, len(train) - 1, size=batch)
        batch_loss = model.step(train[idx], train[idx + 1], lr)
        if s % eval_every == 0:
            history.append((s, batch_loss, model.loss(vx, vy)))
    return TrainResult(history, math.log(vocab_size), count_bigram_loss(train, valid, vocab_size))
