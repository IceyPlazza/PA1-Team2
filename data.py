
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MNIST_DIR = _PROJECT_ROOT / "MNIST"


def _read_idx_images(path: Path) -> np.ndarray:
    """Read an idx3 image file as flattened uint8 rows."""
    raw = np.fromfile(path, dtype=">u4", count=4)
    magic, n, rows, cols = (int(v) for v in raw)
    if magic != 2051:
        raise ValueError(f"{path} is not an idx3 image file (magic {magic})")
    data = np.fromfile(path, dtype=np.uint8, offset=16)
    expected = n * rows * cols
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} pixels, expected {expected}")
    return data.reshape(n, rows * cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """Read an idx1 label file."""
    raw = np.fromfile(path, dtype=">u4", count=2)
    magic, n = (int(v) for v in raw)
    if magic != 2049:
        raise ValueError(f"{path} is not an idx1 label file (magic {magic})")
    labels = np.fromfile(path, dtype=np.uint8, offset=8)
    if labels.size != n:
        raise ValueError(f"{path} has {labels.size} labels, expected {n}")
    return labels.astype(np.int64, copy=False)


def load_mnist(mnist_dir: Path | None = None, dtype: np.dtype = np.float32):
    """Load MNIST train/test sets with pixels scaled to [0, 1]."""
    root = Path(mnist_dir) if mnist_dir is not None else DEFAULT_MNIST_DIR
    train_x = _read_idx_images(root / "train-images.idx3-ubyte")
    train_y = _read_idx_labels(root / "train-labels.idx1-ubyte")
    test_x = _read_idx_images(root / "t10k-images.idx3-ubyte")
    test_y = _read_idx_labels(root / "t10k-labels.idx1-ubyte")

    scale = dtype(255.0)
    train_x = train_x.astype(dtype, copy=False) / scale
    test_x = test_x.astype(dtype, copy=False) / scale
    return train_x, train_y, test_x, test_y
