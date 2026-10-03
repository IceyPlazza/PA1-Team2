"""Part B loss functions on raw logits, each with value() and value_and_grad()."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part_a"))

from mlp import softmax


def _one_hot(y: np.ndarray, n_classes: int, dtype) -> np.ndarray:
    """One-hot encode labels."""
    t = np.zeros((y.shape[0], n_classes), dtype=dtype)
    t[np.arange(y.shape[0]), y] = 1
    return t


class CrossEntropy:
    """Softmax cross-entropy."""

    key = "ce"
    label = "Cross-entropy"

    def value(self, z: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy over the batch."""
        return self.value_and_grad(z, y)[0]

    def value_and_grad(self, z: np.ndarray, y: np.ndarray):
        """Mean cross-entropy and its gradient w.r.t. the logits."""
        p = softmax(z)
        rows = np.arange(y.shape[0])
        loss = float(-np.mean(np.log(np.clip(p[rows, y], 1e-12, 1.0))))
        dz = p
        dz[rows, y] -= 1
        dz /= y.shape[0]
        return loss, dz


class MSE:
    """Squared error between softmax probabilities and one-hot targets."""

    key = "mse"
    label = "MSE"

    def value(self, z: np.ndarray, y: np.ndarray) -> float:
        """Mean squared error over the batch."""
        p = softmax(z)
        diff = p - _one_hot(y, z.shape[1], z.dtype)
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def value_and_grad(self, z: np.ndarray, y: np.ndarray):
        """Mean squared error and its gradient w.r.t. the logits."""
        p = softmax(z)
        diff = p - _one_hot(y, z.shape[1], z.dtype)
        loss = float(np.mean(np.sum(diff * diff, axis=1)))
        g = 2 * diff / y.shape[0]
        dz = p * (g - np.sum(g * p, axis=1, keepdims=True))
        return loss, dz


class MulticlassHinge:
    """Weston-Watkins multiclass hinge loss."""

    key = "hinge"
    label = "Multiclass hinge"

    def _margins(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-class hinge terms max(0, 1 + z_j - z_y)."""
        rows = np.arange(y.shape[0])
        margins = z - z[rows, y][:, None] + 1
        margins[rows, y] = 0
        return np.maximum(margins, 0)

    def value(self, z: np.ndarray, y: np.ndarray) -> float:
        """Mean hinge loss over the batch."""
        return float(np.mean(np.sum(self._margins(z, y), axis=1)))

    def value_and_grad(self, z: np.ndarray, y: np.ndarray):
        """Mean hinge loss and its gradient w.r.t. the logits."""
        margins = self._margins(z, y)
        loss = float(np.mean(np.sum(margins, axis=1)))
        dz = (margins > 0).astype(z.dtype)
        dz[np.arange(y.shape[0]), y] = -dz.sum(axis=1)
        dz /= y.shape[0]
        return loss, dz


LOSSES = {cls.key: cls for cls in (CrossEntropy, MSE, MulticlassHinge)}
