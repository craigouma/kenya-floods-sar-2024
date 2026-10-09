"""Check this pipeline's sigma0 against the Earth Engine sigma0 in Sen1Floods11.

For hand-labelled chips from every event, the original GRD scene is processed
onto the chip's own grid and compared pixel by pixel with the chip: median dB
difference (radiometric bias), correlation of 3x3 means, and the integer shift
that maximises that correlation (geolocation). A bias of more than about 1 dB
or a shift of more than a pixel would put Kenya inputs outside the training
domain of the U-Net.
"""

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from scipy.ndimage import uniform_filter
from shapely.geometry import box, shape

import ancillary
import s1_grd
from config import OUTPUTS, STAC_API
from fetch_sen1floods11 import EVENTS, LOCAL

CHIPS_PER_EVENT = 2
MAX_SHIFT = 3


def chip_grid(path) -> tuple[s1_grd.Grid, np.ndarray, list[float]]:
    with rasterio.open(path) as src:
        grid = s1_grd.Grid(src.crs.to_string(), src.transform, src.width, src.height)
        return grid, src.read().astype("float32"), list(src.bounds)


def scene_for(cat, bounds, date: str, orbit: int):
    geom = box(*bounds)
    items = cat.search(collections=["sentinel-1-grd"], intersects=geom, datetime=date,
                       query={"sat:relative_orbit": {"eq": orbit}}).item_collection()
    items = [i for i in items if shape(i.geometry).contains(geom)]
    return items[0] if items else None


def compare(mine: np.ndarray, gee: np.ndarray) -> dict:
    a, b = uniform_filter(np.nan_to_num(mine, nan=-30), 3), uniform_filter(np.nan_to_num(gee, nan=-30), 3)
    ok = np.isfinite(mine) & np.isfinite(gee)
    m = MAX_SHIFT
    best = (-1.0, 0, 0)
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            sa = np.roll(np.roll(a, dy, 0), dx, 1)[m:-m, m:-m]
            r = np.corrcoef(sa[ok[m:-m, m:-m]], b[m:-m, m:-m][ok[m:-m, m:-m]])[0, 1]
            best = max(best, (r, dy, dx))
    return {"bias_db": float(np.median(mine[ok] - gee[ok])), "r_zero": float(np.corrcoef(a[ok], b[ok])[0, 1]),
            "r_best": best[0], "dy": best[1], "dx": best[2]}


def main() -> None:
    cat = pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)
    meta = pd.read_csv(EVENTS)
    rng = np.random.default_rng(0)
    rows = []
    for ev in meta.itertuples():
        chips = sorted((LOCAL / "HandLabeled" / "S1Hand").glob(f"{ev.chip_prefix}_*_S1Hand.tif"))
        if not chips:
            continue
        for path in rng.choice(chips, size=min(CHIPS_PER_EVENT, len(chips)), replace=False):
            grid, gee, bounds = chip_grid(path)
            item = scene_for(cat, bounds, ev.s1_date.replace("/", "-"), int(ev.rel_orbit_num))
            if item is None:
                rows.append({"chip": path.stem, "event": ev.location, "scene": None})
                continue
            h = ancillary.ellipsoid_height(ancillary.stac_layer("dem", grid), grid)
            mine = s1_grd.sigma0(item, grid, h)
            for i, pol in enumerate(("vv", "vh")):
                rows.append({"chip": path.stem, "event": ev.location, "scene": item.id, "pol": pol,
                             **compare(mine[pol], gee[i])})
            print(rows[-2], rows[-1], sep="\n")
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "grd_vs_gee_sen1floods11.csv", index=False)
    print(df.groupby("pol")[["bias_db", "r_zero", "r_best"]].describe().T.to_string())


if __name__ == "__main__":
    main()
