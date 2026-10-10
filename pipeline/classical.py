"""Classical Sentinel-1 flood mapping baselines.

Thresholds are fitted the way the Sen1Floods11 reference classification was:
Otsu on the pooled histogram of the most heterogeneous 1 km tiles, applied to a
focal-mean smoothed band. Tiles dominated by one class give a unimodal
histogram, on which Otsu's threshold is meaningless.
"""

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu

SMOOTH = 3
TILE = 100
TOP_TILE_FRACTION = 0.1
FIXED_RATIO_DB = -3.0


def smooth(db: np.ndarray) -> np.ndarray:
    valid = np.isfinite(db)
    total = uniform_filter(np.where(valid, db, 0.0), SMOOTH)
    count = uniform_filter(valid.astype(float), SMOOTH)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[~valid] = np.nan
    return out


def tiled_otsu(band: np.ndarray, tile: int = TILE) -> float:
    stds, samples = [], []
    for r in range(0, band.shape[0] - tile + 1, tile):
        for c in range(0, band.shape[1] - tile + 1, tile):
            block = band[r : r + tile, c : c + tile]
            vals = block[np.isfinite(block)]
            if vals.size > 0.9 * tile * tile:
                stds.append(vals.std())
                samples.append(vals)
    if len(samples) < 5:
        vals = band[np.isfinite(band)]
        return float(threshold_otsu(vals)) if vals.size else float("nan")
    keep = np.argsort(stds)[-max(5, int(len(stds) * TOP_TILE_FRACTION)) :]
    return float(threshold_otsu(np.concatenate([samples[i] for i in keep])))


def otsu_water(post_db: np.ndarray, tile: int = TILE) -> tuple[np.ndarray, float]:
    band = smooth(post_db)
    t = tiled_otsu(band, tile)
    return band < t, t


def log_ratio(post_db: np.ndarray, pre_db: np.ndarray) -> np.ndarray:
    return smooth(post_db) - smooth(pre_db)


def change_otsu(post_db: np.ndarray, pre_db: np.ndarray, tile: int = TILE) -> tuple[np.ndarray, float]:
    ratio = log_ratio(post_db, pre_db)
    t = tiled_otsu(ratio, tile)
    return ratio < min(t, 0.0), t


def change_fixed(post_db: np.ndarray, pre_db: np.ndarray) -> np.ndarray:
    return log_ratio(post_db, pre_db) < FIXED_RATIO_DB
