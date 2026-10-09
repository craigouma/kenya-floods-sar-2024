"""Per-AOI input stack: reference, ancillary layers, and pre/post sigma0."""

import sys

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio

import ancillary
import s1_grd
from config import CACHE, REFERENCE, STAC_API
from pick_scenes import SCENES

STACKS = CACHE / "stacks"
STACKS.mkdir(exist_ok=True)


def grid_of(aoi: str) -> s1_grd.Grid:
    with rasterio.open(REFERENCE / f"{aoi}_flood.tif") as src:
        return s1_grd.Grid(src.crs.to_string(), src.transform, src.width, src.height)


def item(item_id: str):
    cat = pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)
    return cat.get_collection("sentinel-1-grd").get_item(item_id)


def stack(aoi: str) -> dict[str, np.ndarray]:
    path = STACKS / f"{aoi}.npz"
    if path.exists():
        return dict(np.load(path))
    grid = grid_of(aoi)
    scene = pd.read_csv(SCENES).set_index("aoi").loc[aoi]
    anc = ancillary.build(aoi, grid)
    with rasterio.open(REFERENCE / f"{aoi}_flood.tif") as src:
        ref = src.read(1)
    out = {"reference": ref, **anc}
    for when in ("post", "pre"):
        s0 = s1_grd.sigma0(item(scene[f"{when}_id"]), grid, anc["ellipsoid_height"])
        out[f"{when}_vv"], out[f"{when}_vh"] = s0["vv"], s0["vh"]
    np.savez_compressed(path, **out)
    return out


if __name__ == "__main__":
    for aoi in sys.argv[1:] or pd.read_csv(SCENES)["aoi"]:
        s = stack(aoi)
        print(aoi, {k: (v.shape, str(v.dtype), float(np.nanmean(v))) for k, v in s.items()})
