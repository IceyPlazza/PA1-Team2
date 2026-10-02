from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# data.py lives at the repo root so other parts can share it.
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplconfig"))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import csv
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import load_mnist
from mlp import MLP, grad_check

PROJECT_ROOT = _ROOT
DEFAULT_OUT = PROJECT_ROOT / "results"
BLAS_THREADS = int(os.environ["VECLIB_MAXIMUM_THREADS"])

WIDTHS = (32, 128, 512)
DEPTHS = (0, 1, 2, 3)
TARGETS = (0.90, 0.95)

DEPTH_COLOR = {0: "#222222", 1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}


def configurations() -> list[tuple[int, int | None]]:
    """(depth, width). Width is None for the depth-0 linear baseline."""
    configs = [(0, None)]
    for depth in DEPTHS:
        if depth == 0:
            continue
        for width in WIDTHS:
            configs.append((depth, width))
    return configs


def layer_sizes(depth: int, width: int | None, n_in: int = 784, n_out: int = 10) -> list[int]:
    if depth == 0:
        return [n_in, n_out]
    if width is None:
        raise ValueError("hidden depth requires a width")
    return [n_in] + [width] * depth + [n_out]


def label_of(depth: int, width: int | None) -> str:
    if depth == 0:
        return "d0 linear"
    return f"d{depth}-w{width}"


def train_one(
    model: MLP,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = train_y.shape[0]
    history = []
    reached = {target: None for target in TARGETS}
    train_seconds = 0.0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        losses = []
        t0 = time.perf_counter()
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            losses.append(model.sgd_step(train_x[batch], train_y[batch], lr))
        train_seconds += time.perf_counter() - t0

        train_acc = model.accuracy(train_x, train_y)
        test_acc = model.accuracy(test_x, test_y)
        mean_loss = float(np.mean(losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_seconds": train_seconds,
            }
        )
        for target in TARGETS:
            if reached[target] is None and test_acc >= target:
                reached[target] = {"epoch": epoch, "train_seconds": train_seconds}
        print(
            f"    epoch {epoch:2d}  loss {mean_loss:.4f}  "
            f"train {train_acc:.4f}  test {test_acc:.4f}  "
            f"train_s {train_seconds:.2f}",
            flush=True,
        )

    final = history[-1]
    row = {
        "train_seconds": final["train_seconds"],
        "epochs": epochs,
        "train_loss": final["train_loss"],
        "train_accuracy": final["train_accuracy"],
        "test_accuracy": final["test_accuracy"],
        "history": history,
    }
    for target in TARGETS:
        tag = str(int(target * 100))
        hit = reached[target]
        row[f"epochs_to_{tag}"] = None if hit is None else hit["epoch"]
        row[f"seconds_to_{tag}"] = None if hit is None else hit["train_seconds"]
    return row


def _annotate(ax, rows, x_key):
    ordered = sorted(rows, key=lambda r: (r[x_key], r["test_accuracy"]))
    for i, row in enumerate(ordered):
        dy = 7 if i % 2 == 0 else -11
        ax.annotate(
            row["name"],
            (row[x_key], row["test_accuracy"] * 100),
            textcoords="offset points",
            xytext=(5, dy),
            fontsize=8,
        )


def plot_accuracy_vs_cost(rows: list[dict], x_key: str, xlabel: str, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for depth in DEPTHS:
        group = [r for r in rows if r["depth"] == depth]
        if not group:
            continue
        ax.scatter(
            [r[x_key] for r in group],
            [r["test_accuracy"] * 100 for r in group],
            s=60,
            color=DEPTH_COLOR[depth],
            label=f"depth {depth}",
            zorder=3,
        )
    _annotate(ax, rows, x_key)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(rows: list[dict], path: Path):
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    colors = plt.cm.tab10.colors
    for i, row in enumerate(rows):
        epochs = [h["epoch"] for h in row["history"]]
        acc = [h["test_accuracy"] * 100 for h in row["history"]]
        ax.plot(
            epochs,
            acc,
            marker="o",
            markersize=3.5,
            color=colors[i % len(colors)],
            label=row["name"],
        )
    ax.axhline(90, color="#888888", linewidth=0.8, linestyle=":", label="90%")
    ax.axhline(95, color="#444444", linewidth=0.8, linestyle="--", label="95%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Test accuracy during training")
    ax.set_xticks(range(1, rows[0]["epochs"] + 1))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(rows: list[dict], path: Path):
    fields = [
        "name",
        "depth",
        "width",
        "layer_sizes",
        "test_accuracy",
        "train_accuracy",
        "train_loss",
        "train_seconds",
        "epochs",
        "epochs_to_90",
        "seconds_to_90",
        "epochs_to_95",
        "seconds_to_95",
        "n_params",
        "nbytes",
        "acc_per_second",
        "acc_per_param",
        "acc_gain_per_extra_second",
        "acc_gain_per_extra_param",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def run(epochs: int, batch_size: int, lr: float, seed: int, out_dir: Path, mnist_dir: Path | None):
    print(f"BLAS threads capped at {BLAS_THREADS}.", flush=True)
    print("Checking backprop against finite differences...", flush=True)
    err = grad_check()
    print(f"Gradient check passed (worst relative error {err:.3e}).", flush=True)

    print("Loading MNIST...", flush=True)
    train_x, train_y, test_x, test_y = load_mnist(mnist_dir)
    print(f"  train {train_x.shape}  test {test_x.shape}  dtype {train_x.dtype}", flush=True)

    # Touch BLAS once so the first (tiny) model is not charged for library warmup.
    warmup = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    _ = warmup @ warmup

    rows = []
    for depth, width in configurations():
        sizes = layer_sizes(depth, width)
        name = label_of(depth, width)
        model = MLP(sizes, seed=seed, dtype=np.float32)
        print(
            f"\n{name}  sizes={sizes}  params={model.n_params:,}  bytes={model.nbytes:,}",
            flush=True,
        )
        trained = train_one(
            model,
            train_x,
            train_y,
            test_x,
            test_y,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed + 1,
        )
        row = {
            "name": name,
            "depth": depth,
            "width": width if width is not None else "",
            "layer_sizes": "x".join(str(s) for s in sizes),
            "n_params": model.n_params,
            "nbytes": model.nbytes,
            **trained,
        }
        rows.append(row)

    baseline = rows[0]
    for row in rows:
        row["acc_per_second"] = row["test_accuracy"] / row["train_seconds"]
        row["acc_per_param"] = row["test_accuracy"] / row["n_params"]
        extra_seconds = row["train_seconds"] - baseline["train_seconds"]
        extra_params = row["n_params"] - baseline["n_params"]
        gain = row["test_accuracy"] - baseline["test_accuracy"]
        row["acc_gain_per_extra_second"] = None if extra_seconds <= 0 else gain / extra_seconds
        row["acc_gain_per_extra_param"] = None if extra_params <= 0 else gain / extra_params

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "sweep_results.csv")

    plot_accuracy_vs_cost(
        rows,
        "train_seconds",
        "Training time (seconds)",
        "Accuracy vs training time",
        out_dir / "accuracy_vs_time.png",
    )
    plot_accuracy_vs_cost(
        rows,
        "n_params",
        "Parameter count",
        "Accuracy vs model size",
        out_dir / "accuracy_vs_params.png",
    )
    plot_learning_curves(rows, out_dir / "learning_curves.png")
    print(f"Wrote {out_dir / 'sweep_results.csv'}")
    print(f"Wrote plots in {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Part A architecture sweep on MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mnist-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.epochs, args.batch_size, args.lr, args.seed, args.out, args.mnist_dir)
