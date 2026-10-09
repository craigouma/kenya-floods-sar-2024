"""Flood-class agreement metrics and the spatial block bootstrap."""

import numpy as np

from config import N_BOOTSTRAP, SEED


def counts(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> np.ndarray:
    p, r = pred[valid].astype(bool), ref[valid].astype(bool)
    return np.array([(p & r).sum(), (p & ~r).sum(), (~p & r).sum(), (~p & ~r).sum()], dtype="int64")


def scores(c: np.ndarray) -> dict:
    c = np.asarray(c, dtype=float)
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "iou": tp / (tp + fp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn),
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
        }


def block_counts(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray, block: int) -> np.ndarray:
    """Confusion counts per square block of `block` pixels, shape (n_blocks, 4)."""
    h, w = pred.shape
    out = []
    for r in range(0, h, block):
        for c in range(0, w, block):
            v = valid[r : r + block, c : c + block]
            if v.any():
                out.append(counts(pred[r : r + block, c : c + block], ref[r : r + block, c : c + block], v))
    return np.array(out).reshape(-1, 4)


def bootstrap(blocks: np.ndarray, n: int = N_BOOTSTRAP, seed: int = SEED) -> dict[str, tuple[float, float]]:
    """95% intervals from resampling 1 km blocks with replacement.

    Pixels inside a block are spatially autocorrelated, so the block, not the
    pixel, is the resampling unit.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n, len(blocks)))
    sums = blocks[idx].sum(axis=1)
    s = scores(sums)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in s.items()}
