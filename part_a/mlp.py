
from __future__ import annotations
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, layer_sizes: list[int], seed: int = 0, dtype=np.float32):
        if len(layer_sizes) < 2:
            raise ValueError("need at least an input size and an output size")
        self.sizes = [int(s) for s in layer_sizes]
        self.dtype = np.dtype(dtype)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []

        rng = np.random.default_rng(seed)
        n_layers = len(self.sizes) - 1
        for i, (n_in, n_out) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            if i < n_layers - 1:
                std = np.sqrt(2.0 / n_in)
            else:
                std = np.sqrt(2.0 / (n_in + n_out))
            weights = rng.standard_normal((n_in, n_out)) * std
            self.W.append(weights.astype(self.dtype, copy=False))
            self.b.append(np.zeros(n_out, dtype=self.dtype))

    @property
    def n_hidden(self) -> int:
        return len(self.sizes) - 2

    @property
    def n_params(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    @property
    def nbytes(self) -> int:
        return int(sum(w.nbytes + b.nbytes for w, b in zip(self.W, self.b)))

    def forward(self, x: np.ndarray):
        activations = [x]
        pre: list[np.ndarray] = []
        last = len(self.W) - 1
        a = x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            pre.append(z)
            if i < last:
                a = np.maximum(z, 0)
            else:
                a = softmax(z)
            activations.append(a)
        return a, activations, pre

    def loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        clipped = np.clip(probs, 1e-12, 1.0)
        chosen = clipped[np.arange(y.shape[0]), y]
        return float(-np.mean(np.log(chosen)))

    def backward(self, activations, pre, y: np.ndarray):
        probs = activations[-1]
        batch = y.shape[0]
        dz = probs.copy()
        dz[np.arange(batch), y] -= self.dtype.type(1)
        dz /= self.dtype.type(batch)

        grad_w = [None] * len(self.W)
        grad_b = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            a_prev = activations[i]
            grad_w[i] = a_prev.T @ dz
            grad_b[i] = dz.sum(axis=0)
            if i > 0:
                da = dz @ self.W[i].T
                dz = da * (pre[i - 1] > 0)
        return grad_w, grad_b

    def sgd_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        probs, activations, pre = self.forward(x)
        loss = self.loss(probs, y)
        grad_w, grad_b = self.backward(activations, pre, y)
        step = self.dtype.type(lr)
        for w, b, dw, db in zip(self.W, self.b, grad_w, grad_b):
            w -= step * dw
            b -= step * db
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs, _, _ = self.forward(x)
        return probs.argmax(axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray, batch_size: int = 2048) -> float:
        correct = 0
        for start in range(0, y.shape[0], batch_size):
            stop = start + batch_size
            pred = self.predict(x[start:stop])
            correct += int(np.sum(pred == y[start:stop]))
        return correct / y.shape[0]


def grad_check(eps: float = 1e-5, rtol: float = 1e-5) -> float:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 5)).astype(np.float64)
    y = rng.integers(0, 3, size=6)
    model = MLP([5, 4, 3], seed=1, dtype=np.float64)
    probs, activations, pre = model.forward(x)
    grad_w, grad_b = model.backward(activations, pre, y)

    worst = 0.0
    for i, (w, gw) in enumerate(zip(model.W, grad_w)):
        for index in np.ndindex(w.shape):
            original = w[index]
            w[index] = original + eps
            loss_plus = model.loss(model.forward(x)[0], y)
            w[index] = original - eps
            loss_minus = model.loss(model.forward(x)[0], y)
            w[index] = original
            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytic = float(gw[index])
            denom = abs(numerical) + abs(analytic) + 1e-12
            err = abs(numerical - analytic) / denom
            worst = max(worst, err)
        b = model.b[i]
        gb = grad_b[i]
        for j in range(b.shape[0]):
            original = b[j]
            b[j] = original + eps
            loss_plus = model.loss(model.forward(x)[0], y)
            b[j] = original - eps
            loss_minus = model.loss(model.forward(x)[0], y)
            b[j] = original
            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytic = float(gb[j])
            denom = abs(numerical) + abs(analytic) + 1e-12
            err = abs(numerical - analytic) / denom
            worst = max(worst, err)

    if worst > rtol:
        raise AssertionError(f"gradient check failed: worst relative error {worst:.3e}")
    return worst
