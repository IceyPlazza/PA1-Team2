"""Part B: train the same network under each loss and record learning curves and cost-to-target.

Every run shares the architecture, initial weights, shuffle order, learning rate, batch size and
epoch budget, so the loss is the only variable. Test accuracy is sampled every --eval-every SGD
steps (not just per epoch) so steps / epochs / seconds to a target can be resolved below one epoch.
Wall-clock time counts SGD steps only; evaluation is excluded, matching Part A.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "part_a"))
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplconfig"))
# Same BLAS thread caps as Part A so timings are comparable across parts.
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
from losses import LOSSES, CrossEntropy
from mlp import MLP, grad_check

DEFAULT_OUT = _ROOT / "results" / "part_b"
DEFAULT_SIZES = "784,128,128,10"
BLAS_THREADS = int(os.environ["VECLIB_MAXIMUM_THREADS"])
TARGETS = (0.90,)

# Each loss keeps one color in every plot (categorical slots 1-3, CVD-checked as a set).
LOSS_COLOR = {"ce": "#2a78d6", "mse": "#eb6834", "hinge": "#1baf7a"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def tag(target: float) -> str:
    return str(int(round(target * 100)))


def evaluate(model: MLP, x: np.ndarray, y: np.ndarray, loss_fn, batch_size: int = 2048):
    """Return (accuracy, mean objective under loss_fn, mean cross-entropy) over a dataset."""
    ce = CrossEntropy()
    correct, objective, cross_entropy = 0, 0.0, 0.0
    for start in range(0, y.shape[0], batch_size):
        xb, yb = x[start : start + batch_size], y[start : start + batch_size]
        z = model.forward(xb)[2][-1]
        correct += int(np.sum(z.argmax(axis=1) == yb))
        objective += loss_fn.value(z, yb) * yb.shape[0]
        cross_entropy += ce.value(z, yb) * yb.shape[0]
    n = y.shape[0]
    return correct / n, objective / n, cross_entropy / n


def train(
    model: MLP, loss_fn, data, epochs: int, batch_size: int, lr: float, seed: int, eval_every: int,
    verbose: bool = True,
):
    train_x, train_y, test_x, test_y = data
    rng = np.random.default_rng(seed)
    n = train_y.shape[0]
    steps_per_epoch = -(-n // batch_size)
    step = 0
    train_seconds = 0.0
    checkpoints = [
        {"step": 0, "epoch": 0.0, "train_seconds": 0.0, "test_accuracy": model.accuracy(test_x, test_y)}
    ]
    history = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            t0 = time.perf_counter()
            losses.append(model.sgd_step(train_x[batch], train_y[batch], lr, loss_fn))
            train_seconds += time.perf_counter() - t0
            step += 1
            if step % eval_every == 0 or step == epoch * steps_per_epoch:
                checkpoints.append(
                    {
                        "step": step,
                        "epoch": step / steps_per_epoch,
                        "train_seconds": train_seconds,
                        "test_accuracy": model.accuracy(test_x, test_y),
                    }
                )

        test_acc, test_objective, test_ce = evaluate(model, test_x, test_y, loss_fn)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": model.accuracy(train_x, train_y),
            "test_accuracy": test_acc,
            "test_loss": test_objective,
            "test_cross_entropy": test_ce,
            "train_seconds": train_seconds,
        }
        history.append(record)
        if not verbose:
            continue
        print(
            f"    epoch {epoch:2d}  loss {record['train_loss']:.4f}  "
            f"train {record['train_accuracy']:.4f}  test {test_acc:.4f}  "
            f"test_ce {test_ce:.4f}  train_s {train_seconds:.2f}",
            flush=True,
        )
    return history, checkpoints, steps_per_epoch


def merge_repeats(reps: list[tuple[list[dict], list[dict], int]]):
    """Keep the first repeat's curves and replace every train_seconds with the median across
    repeats. Returns (history, checkpoints, identical) where identical says whether all repeats
    produced the same accuracy trajectory (expected, since seeds are fixed)."""
    history = [dict(h) for h in reps[0][0]]
    checkpoints = [dict(c) for c in reps[0][1]]
    for records, index in ((history, 0), (checkpoints, 1)):
        for i, record in enumerate(records):
            record["train_seconds"] = float(np.median([rep[index][i]["train_seconds"] for rep in reps]))
    identical = all(
        [c["test_accuracy"] for c in rep[1]] == [c["test_accuracy"] for c in reps[0][1]] for rep in reps
    )
    return history, checkpoints, identical


def summarize(key: str, reps: list[tuple[list[dict], list[dict], int]]) -> dict:
    history, checkpoints, identical = merge_repeats(reps)
    steps_per_epoch = reps[0][2]
    totals = [rep[0][-1]["train_seconds"] for rep in reps]
    final = history[-1]
    row = {
        "loss": key,
        "label": LOSSES[key].label,
        "final_test_accuracy": final["test_accuracy"],
        "final_train_accuracy": final["train_accuracy"],
        "best_test_accuracy": max(c["test_accuracy"] for c in checkpoints),
        "final_test_cross_entropy": final["test_cross_entropy"],
        "train_seconds": final["train_seconds"],
        "seconds_per_epoch": final["train_seconds"] / len(history),
        "ms_per_step": 1e3 * final["train_seconds"] / (len(history) * steps_per_epoch),
        "steps_per_epoch": steps_per_epoch,
        "repeats": len(reps),
        "train_seconds_min": min(totals),
        "train_seconds_max": max(totals),
        "repeats_identical": identical,
    }
    for target in TARGETS:
        t = tag(target)
        hit = next((c for c in checkpoints if c["test_accuracy"] >= target), None)
        full = next((h["epoch"] for h in history if h["test_accuracy"] >= target), None)
        row[f"steps_to_{t}"] = None if hit is None else hit["step"]
        row[f"epochs_to_{t}"] = None if hit is None else round(hit["epoch"], 3)
        row[f"seconds_to_{t}"] = None if hit is None else hit["train_seconds"]
        row[f"full_epochs_to_{t}"] = full
    return row, history, checkpoints


def add_cost_vs_ce(rows: list[dict]):
    """steps_vs_ce_T = steps this loss needed to reach T divided by what cross-entropy needed."""
    ce = next((r for r in rows if r["loss"] == "ce"), None)
    for row in rows:
        for target in TARGETS:
            t = tag(target)
            mine = row[f"steps_to_{t}"]
            base = None if ce is None else ce[f"steps_to_{t}"]
            row[f"steps_vs_ce_{t}"] = None if mine is None or not base else round(mine / base, 3)


def write_rows(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_accuracy(runs: list[dict], x_key: str, xlabel: str, title: str, path: Path):
    """Left panel: whole run on a log x-axis, since most of the climb happens inside epoch 1.
    Right panel: linear x, zoomed onto the target band."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
    x_max = max(run["checkpoints"][-1][x_key] for run in runs)
    for ax, zoomed in zip(axes, (False, True)):
        _style(ax)
        for run in runs:
            # Log axis cannot show the step-0 checkpoint.
            points = run["checkpoints"] if zoomed else run["checkpoints"][1:]
            ax.plot(
                [c[x_key] for c in points], [c["test_accuracy"] * 100 for c in points],
                color=LOSS_COLOR[run["key"]], linewidth=1.6, label=LOSSES[run["key"]].label,
            )
        for target in TARGETS:
            ax.axhline(target * 100, color=INK_2, linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
            if zoomed:
                ax.annotate(
                    f"{tag(target)}%", (0, target * 100), xytext=(3, 2), textcoords="offset points",
                    fontsize=7, color=INK_2, xycoords=("axes fraction", "data"),
                )
        if zoomed:
            ax.set_ylim(85, 100)
            ax.set_xlim(0, x_max)
        else:
            ax.set_ylim(0, 100)
            ax.set_xscale("log")
            ax.set_xlim(right=x_max)
        ax.set_xlabel(xlabel + (" (log scale)" if not zoomed else ""))
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    axes[0].set_title("Full run", fontsize=10, loc="left")
    axes[1].set_title("Zoomed to 85-100% (target 90%)", fontsize=10, loc="left")
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_losses(runs: list[dict], path: Path):
    """One panel per loss (each objective has its own scale), plus test cross-entropy for every
    run on a shared axis as the common yardstick."""
    fig, axes = plt.subplots(1, len(runs) + 1, figsize=(3.6 * (len(runs) + 1), 3.8), facecolor=SURFACE)
    for ax, run in zip(axes, runs):
        _style(ax)
        epochs = [h["epoch"] for h in run["history"]]
        color = LOSS_COLOR[run["key"]]
        ax.plot(epochs, [h["train_loss"] for h in run["history"]], color=color, linewidth=1.6, label="train")
        ax.plot(
            epochs, [h["test_loss"] for h in run["history"]], color=color, linewidth=1.6,
            linestyle=(0, (4, 2)), label="test",
        )
        ax.set_title(f"{LOSSES[run['key']].label} objective", fontsize=10, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax = axes[-1]
    _style(ax)
    for run in runs:
        ax.plot(
            [h["epoch"] for h in run["history"]], [h["test_cross_entropy"] for h in run["history"]],
            color=LOSS_COLOR[run["key"]], linewidth=1.6, label=LOSSES[run["key"]].label,
        )
    ax.set_title("Test cross-entropy (all runs)", fontsize=10, loc="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run(args):
    sizes = [int(s) for s in args.sizes.split(",")]
    keys = [k.strip() for k in args.losses.split(",")]
    unknown = [k for k in keys if k not in LOSSES]
    if unknown:
        raise SystemExit(f"unknown loss(es) {unknown}; choose from {list(LOSSES)}")

    print(f"BLAS threads capped at {BLAS_THREADS}.", flush=True)
    print("Checking backprop against finite differences...", flush=True)
    for key in keys:
        err = grad_check(loss_fn=LOSSES[key]())
        print(f"  {key:6s} worst relative error {err:.3e}", flush=True)

    print("Loading MNIST...", flush=True)
    data = load_mnist(args.mnist_dir)
    print(f"  train {data[0].shape}  test {data[2].shape}  dtype {data[0].dtype}", flush=True)

    # Touch BLAS once so the first run is not charged for library warmup.
    warmup = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    _ = warmup @ warmup

    # Rotate the loss order each repeat so no loss is always run first (or last) on a warming CPU.
    reps = {key: [] for key in keys}
    for rep in range(args.repeats):
        shift = rep % len(keys)
        for key in keys[shift:] + keys[:shift]:
            model = MLP(sizes, seed=args.seed, dtype=np.float32)
            print(
                f"\n[repeat {rep + 1}/{args.repeats}] {LOSSES[key].label}  sizes={sizes}  "
                f"params={model.n_params:,}",
                flush=True,
            )
            result = train(
                model, LOSSES[key](), data, args.epochs, args.batch_size, args.lr, args.seed + 1,
                args.eval_every, verbose=rep == 0,
            )
            reps[key].append(result)
            if rep > 0:
                print(f"    train_s {result[0][-1]['train_seconds']:.2f}", flush=True)

    runs, rows, epoch_rows, checkpoint_rows = [], [], [], []
    for key in keys:
        row, history, checkpoints = summarize(key, reps[key])
        if not row["repeats_identical"]:
            print(f"WARNING: {key} accuracy differed across repeats; curves show repeat 1.", flush=True)
        runs.append({"key": key, "history": history, "checkpoints": checkpoints})
        rows.append(row)
        epoch_rows += [{"loss": key, **h} for h in history]
        checkpoint_rows += [{"loss": key, **c} for c in checkpoints]

    add_cost_vs_ce(rows)
    common = {
        "layer_sizes": "x".join(str(s) for s in sizes),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "eval_every": args.eval_every,
    }
    for row in rows:
        row.update(common)

    target_fields = [
        f"{prefix}_{tag(t)}"
        for t in TARGETS
        for prefix in ("steps_to", "epochs_to", "seconds_to", "full_epochs_to", "steps_vs_ce")
    ]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_rows(
        out / "loss_summary.csv",
        rows,
        ["loss", "label", "final_test_accuracy", "final_train_accuracy", "best_test_accuracy",
         "final_test_cross_entropy", "train_seconds", "train_seconds_min", "train_seconds_max",
         "seconds_per_epoch", "ms_per_step", "steps_per_epoch", *target_fields, "repeats",
         "repeats_identical", *common],
    )
    write_rows(out / "loss_epoch_history.csv", epoch_rows, ["loss", *history[0].keys()])
    write_rows(out / "loss_checkpoints.csv", checkpoint_rows, ["loss", *checkpoints[0].keys()])

    plot_accuracy(runs, "epoch", "Epoch", "Test accuracy vs epoch, by loss", out / "accuracy_vs_epoch.png")
    plot_accuracy(
        runs, "train_seconds", "Training time (seconds, SGD only)", "Test accuracy vs training time, by loss",
        out / "accuracy_vs_time.png",
    )
    plot_losses(runs, out / "loss_vs_epoch.png")

    print("\nCost to reach target test accuracy (steps / epochs / seconds):")
    for row in rows:
        cells = []
        for t in TARGETS:
            s = row[f"steps_to_{tag(t)}"]
            cells.append(
                f"{tag(t)}%: never" if s is None
                else f"{tag(t)}%: {s} / {row[f'epochs_to_{tag(t)}']:.2f} / {row[f'seconds_to_{tag(t)}']:.2f}s"
            )
        print(f"  {row['label']:17s} final {row['final_test_accuracy']:.4f}   " + "   ".join(cells))
    print(f"\nWrote CSVs and plots in {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Part B loss-function comparison on MNIST")
    parser.add_argument("--losses", default=",".join(LOSSES), help=f"comma list from {list(LOSSES)}")
    parser.add_argument("--sizes", default=DEFAULT_SIZES, help="layer sizes; default is Part A's d2-w128")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50, help="SGD steps between test-accuracy samples")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats per loss; seconds are medians")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mnist-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
