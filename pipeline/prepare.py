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


def items(pass_id: str) -> list:
    cat = pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)
    collection = cat.get_collection("sentinel-1-grd")
    return [collection.get_item(i) for i in pass_id.split(";")]


def stack(aoi: str, post_id: str | None = None, pre_id: str | None = None) -> dict[str, np.ndarray]:
    """Input stack for an AOI, with the scenes from scenes.csv unless overridden
    (the day-gap sensitivity analysis swaps in other passes)."""
    scene = pd.read_csv(SCENES).set_index("aoi").loc[aoi]
    ids = {"post": post_id or scene["post_id"], "pre": pre_id or scene["pre_id"]}
    tag = aoi if post_id is None and pre_id is None else f"{aoi}__{ids['post'][17:25]}__{ids['pre'][17:25]}"
    path = STACKS / f"{tag}.npz"
    if path.exists():
        return dict(np.load(path))
    grid = grid_of(aoi)
    anc = ancillary.build(aoi, grid)
    with rasterio.open(REFERENCE / f"{aoi}_flood.tif") as src:
        ref = src.read(1)
    out = {"reference": ref, **anc}
    for when in ("post", "pre"):
        s0 = s1_grd.sigma0_pass(items(ids[when]), grid, anc["ellipsoid_height"])
        out[f"{when}_vv"], out[f"{when}_vh"] = s0["vv"], s0["vh"]
    np.savez_compressed(path, **out)
    return out


if __name__ == "__main__":
    for aoi in sys.argv[1:] or pd.read_csv(SCENES)["aoi"]:
        s = stack(aoi)
        print(aoi, {k: (v.shape, str(v.dtype), float(np.nanmean(v))) for k, v in s.items()})
