"""Loss functions for the Part B comparison.

Every loss takes the output layer's raw logits z (batch x classes) and integer labels y, and exposes
    value(z, y)          -> mean loss over the batch
    value_and_grad(z, y) -> (mean loss, dLoss/dz)
so MLP.sgd_step / MLP.backward_from can train with any of them unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part_a"))

from mlp import softmax


def _one_hot(y: np.ndarray, n_classes: int, dtype) -> np.ndarray:
    t = np.zeros((y.shape[0], n_classes), dtype=dtype)
    t[np.arange(y.shape[0]), y] = 1
    return t


class CrossEntropy:
    """Softmax + negative log-likelihood. Gradient (p - onehot) / B never shrinks while the
    model is wrong, which is why it converges fastest."""

    key = "ce"
    label = "Cross-entropy"

    def value(self, z: np.ndarray, y: np.ndarray) -> float:
        return self.value_and_grad(z, y)[0]

    def value_and_grad(self, z: np.ndarray, y: np.ndarray):
        p = softmax(z)
        rows = np.arange(y.shape[0])
        loss = float(-np.mean(np.log(np.clip(p[rows, y], 1e-12, 1.0))))
        dz = p
        dz[rows, y] -= 1
        dz /= y.shape[0]
        return loss, dz


class MSE:
    """Squared error between softmax probabilities and one-hot targets, summed over classes and
    averaged over the batch. The gradient passes through the softmax Jacobian
    dz = p * (g - sum(g * p)), which vanishes as p saturates even when the prediction is wrong."""

    key = "mse"
    label = "MSE"

    def value(self, z: np.ndarray, y: np.ndarray) -> float:
        p = softmax(z)
        diff = p - _one_hot(y, z.shape[1], z.dtype)
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def value_and_grad(self, z: np.ndarray, y: np.ndarray):
        p = softmax(z)
        diff = p - _one_hot(y, z.shape[1], z.dtype)
        loss = float(np.mean(np.sum(diff * diff, axis=1)))
        g = 2 * diff / y.shape[0]
        dz = p * (g - np.sum(g * p, axis=1, keepdims=True))
        return loss, dz


class MulticlassHinge:
    """Weston-Watkins multiclass hinge on raw logits: sum over j != y of max(0, 1 + z_j - z_y).
    Gradient is a constant +-1 per violated margin and exactly 0 once every margin is met."""

    key = "hinge"
    label = "Multiclass hinge"

    def _margins(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        rows = np.arange(y.shape[0])
        margins = z - z[rows, y][:, None] + 1
        margins[rows, y] = 0
        return np.maximum(margins, 0)

    def value(self, z: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(np.sum(self._margins(z, y), axis=1)))

    def value_and_grad(self, z: np.ndarray, y: np.ndarray):
        margins = self._margins(z, y)
        loss = float(np.mean(np.sum(margins, axis=1)))
        dz = (margins > 0).astype(z.dtype)
        dz[np.arange(y.shape[0]), y] = -dz.sum(axis=1)
        dz /= y.shape[0]
        return loss, dz


LOSSES = {cls.key: cls for cls in (CrossEntropy, MSE, MulticlassHinge)}
