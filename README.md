# PA1: Train, Sweep & Study Training Cost from Scratch

This project trains MNIST digit classifiers from scratch using only NumPy (no deep-learning frameworks) and measures what
training costs:

- **Part A** sweeps network depth (0–3 hidden layers) and width (32, 128 and 512 neurons). It records accuracy,
  training time and model size for each configuration.
- **Part B** keeps one architecture fixed (784-128-128-10) and compares cross-entropy, MSE and multiclass hinge loss
  by how many steps, epochs and seconds each one needs to reach 90% test accuracy.

## Setup

You need Python 3.11 or newer.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

The MNIST files are already in `MNIST/`, so you don't need to download anything. To use a copy stored somewhere else,
pass `--mnist-dir <folder>`. That folder must contain `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`,
`t10k-images.idx3-ubyte` and `t10k-labels.idx1-ubyte`.

## Running

Run every command from the repo root.

### Part A: architecture sweep

```bash
python part_a/sweep.py
```

This trains all 10 configurations (d0 linear, then d1–d3 at each width) for 10 epochs each, with batch size 128,
learning rate 0.1 and seed 0. Before training, it checks backprop against finite differences.

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 10 | Epochs per configuration |
| `--batch-size` | 128 | Mini-batch size |
| `--lr` | 0.1 | SGD learning rate |
| `--seed` | 0 | Seed for weight initialization (the shuffle uses `seed + 1`) |
| `--out` | `results/` | Output folder |
| `--mnist-dir` | `MNIST/` | Folder containing the MNIST idx files |
| `--plot-only` | off | Redraw the two accuracy-vs-cost plots from the existing `sweep_results.csv` without training |

The script writes these files to `results/`:

| File | Contents |
|---|---|
| `sweep_results.csv` | One row per configuration: accuracy, training time, epochs and seconds to 90%/95%, params, bytes, accuracy per second, accuracy per parameter |
| `accuracy_vs_time.png` | Test accuracy vs training time, with the Pareto frontier |
| `accuracy_vs_params.png` | Test accuracy vs parameter count, with the Pareto frontier |
| `learning_curves.png` | Test accuracy per epoch for every configuration |

### Part B: loss-function comparison

```bash
python part_b/loss_compare.py
```

This trains the 784-128-128-10 network (Part A's d2-w128) once per loss, for 20 epochs, and repeats the whole set
3 times. Every run starts from the same initial weights and uses the same shuffle order, so the loss is the only thing
that changes. Test accuracy is sampled every 50 SGD steps. Reported times are the median across repeats and count only
the SGD steps, not evaluation.

| Option | Default | Meaning |
|---|---|---|
| `--losses` | `ce,mse,hinge` | Comma-separated losses to compare |
| `--sizes` | `784,128,128,10` | Layer sizes |
| `--epochs` | 20 | Epochs per run |
| `--batch-size` | 128 | Mini-batch size |
| `--lr` | 0.1 | SGD learning rate |
| `--seed` | 0 | Seed for weight initialization (the shuffle uses `seed + 1`) |
| `--eval-every` | 50 | SGD steps between test-accuracy samples |
| `--repeats` | 3 | Timed repeats per loss (the loss order rotates each repeat) |
| `--out` | `results/part_b/` | Output folder |
| `--mnist-dir` | `MNIST/` | Folder containing the MNIST idx files |

The script writes these files to `results/part_b/`:

| File | Contents |
|---|---|
| `loss_summary.csv` | One row per loss: final and best accuracy, steps/epochs/seconds to 90%, timing |
| `loss_epoch_history.csv` | Per-epoch train/test loss and accuracy for each loss |
| `loss_checkpoints.csv` | Test accuracy every `--eval-every` steps for each loss |
| `accuracy_vs_epoch.png` | Test accuracy vs epoch (full run and zoomed) |
| `accuracy_vs_time.png` | Test accuracy vs SGD training time (full run and zoomed) |
| `loss_vs_epoch.png` | Train/test objective per loss, plus test cross-entropy for all runs |

### Quick smoke test

This runs both parts quickly and writes to a separate folder, so the committed results aren't overwritten:

```bash
python part_a/sweep.py --epochs 1 --out smoke/part_a
python part_b/loss_compare.py --epochs 1 --repeats 1 --out smoke/part_b
```

## Code layout

| Path | Purpose |
|---|---|
| `data.py` | Reads the MNIST idx files and scales pixels to [0, 1] |
| `part_a/mlp.py` | NumPy MLP (ReLU hidden layers, softmax output), SGD step, gradient check |
| `part_a/sweep.py` | Part A depth/width sweep |
| `part_b/losses.py` | Cross-entropy, MSE-on-softmax and Weston-Watkins multiclass hinge losses with gradients |
| `part_b/loss_compare.py` | Part B loss comparison |
| `results/` | Committed outputs used in the report |

## Notes on timing

- Wall-clock times depend on the machine and can drift between runs. The scripts limit BLAS to 4 threads and warm it up
  before the first timed run, but if you rerun them, expect different seconds while the accuracies stay identical.
- Part B repeats every run and reports median times because of this drift. Treat steps and epochs to target as the
  main cost measure and seconds as secondary.
- With fixed seeds, every repeat produces exactly the same accuracy curve. If one doesn't, `loss_compare.py` prints a
  warning.
