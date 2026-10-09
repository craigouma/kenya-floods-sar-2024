"""Pre-event Sentinel-1 chips for the Sen1Floods11 hand-labelled set.

Sen1Floods11 ships only the flood-date image. For each hand-labelled chip the
latest acquisition from the same relative orbit at least MIN_LEAD_DAYS before
the event is processed onto the chip grid with the same GRD pipeline used for
Kenya, which validate_grd.py shows matches the Earth Engine chips to within a
few hundredths of a dB.
"""

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from shapely.geometry import box, shape
from shapely.ops import unary_union

import ancillary
import s1_grd
from config import CACHE, DATA, STAC_API
from fetch_sen1floods11 import EVENTS, LOCAL

MIN_LEAD_DAYS = 30
SEARCH_DAYS = 120
OUT = CACHE / "sen1floods11" / "HandLabeled" / "S1PreHand"
INDEX = DATA / "sen1floods11_pre_event.csv"


def pre_scenes(cat, geom, event_date: pd.Timestamp, orbit: int) -> list:
    """Slices of the latest qualifying pass that together cover the chip; a
    chip can straddle two consecutive slices of one datatake."""
    end = event_date - pd.Timedelta(days=MIN_LEAD_DAYS)
    start = end - pd.Timedelta(days=SEARCH_DAYS)
    items = cat.search(collections=["sentinel-1-grd"], intersects=geom,
                       datetime=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                       query={"sat:relative_orbit": {"eq": orbit}}).item_collection()
    passes = {}
    for i in items:
        if {"vv", "vh"} <= set(i.assets):
            passes.setdefault(i.datetime.date(), []).append(i)
    for day in sorted(passes, reverse=True):
        if unary_union([shape(i.geometry) for i in passes[day]]).contains(geom):
            return passes[day]
    return []


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cat = pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)
    events = pd.read_csv(EVENTS).set_index("chip_prefix")
    rows = []
    for path in sorted((LOCAL / "HandLabeled" / "S1Hand").glob("*_S1Hand.tif")):
        target = OUT / path.name.replace("S1Hand", "S1PreHand")
        ev = events.loc[path.name.split("_")[0]]
        with rasterio.open(path) as src:
            grid = s1_grd.Grid(src.crs.to_string(), src.transform, src.width, src.height)
            profile = src.profile
            bounds = src.bounds
        items = pre_scenes(cat, box(*bounds), pd.Timestamp(ev.s1_date.replace("/", "-")), int(ev.rel_orbit_num))
        row = {"chip": path.stem.removesuffix("_S1Hand"), "event_date": ev.s1_date,
               "pre_id": ";".join(i.id for i in items) or None,
               "pre_date": items[0].datetime.date().isoformat() if items else None}
        if items and not target.exists():
            h = ancillary.ellipsoid_height(ancillary.stac_layer("dem", grid), grid)
            stack = np.full((2, grid.height, grid.width), np.nan, dtype="float32")
            for item in items:
                try:
                    s0 = s1_grd.sigma0(item, grid, h)
                except ValueError:
                    continue
                for b, pol in enumerate(("vv", "vh")):
                    stack[b] = np.where(np.isnan(stack[b]), s0[pol], stack[b])
            with rasterio.open(target, "w", **{**profile, "count": 2, "dtype": "float32", "compress": "deflate"}) as dst:
                dst.write(stack)
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(INDEX, index=False)


if __name__ == "__main__":
    main()
