"""Lower Tana flood extent through the 2024 long rains.

Applies one method to every Sentinel-1 pass from April to June 2024 over the
corridor tiles in Garissa and Tana River counties, plus the Garissa extent.
There is no reference for most of this area; the riverine accuracy from the
evaluation is its expected error. Each relative orbit images a different part
of the corridor, so areas are reported per orbit with the observed area
alongside, and are only comparable within an orbit.
"""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.features import rasterize

import ancillary
import classical
import s1_grd
from boundaries import counties
from config import (CACHE, MODELS, OUTPUTS, PERMANENT_WATER_OCCURRENCE, PIXEL_SIZE, PRE_EVENT_END,
                    PRE_EVENT_START)
from methods import UNETS
from pick_scenes import catalog, covering
from prepare import grid_of, items
from reference import SPEC, UTM, extent
from unet import UNet, predict

START, END = "2024-04-01", "2024-06-30"
COUNTIES = {"KE007": 1, "KE004": 2}
COUNTY_NAMES = {1: "Garissa", 2: "Tana River"}
OUT = OUTPUTS / "tana"
PRE = CACHE / "tana_pre"
PX_KM2 = PIXEL_SIZE**2 / 1e6


def region_aois() -> list[str]:
    region = counties().set_index("pcode").loc[list(COUNTIES)].to_crs(UTM).union_all()
    return [a for a, s in SPEC.items() if (a.startswith("tana_") or a == "garissa_s2") and extent(s).intersects(region)]


def county_raster(grid) -> np.ndarray:
    c = counties().set_index("pcode").loc[list(COUNTIES)].to_crs(UTM)
    return rasterize(((g, COUNTIES[p]) for p, g in c.geometry.items()), out_shape=(grid.height, grid.width),
                     transform=grid.transform, fill=0, dtype="uint8")


def flood_mask(method: str, post: dict, pre: dict) -> np.ndarray:
    pol = method.rsplit("_", 1)[-1]
    if method.startswith("change_fixed"):
        return classical.change_fixed(post[pol], pre[pol])
    if method.startswith("change_otsu"):
        return classical.change_otsu(post[pol], pre[pol])[0]
    if method.startswith("otsu"):
        return classical.otsu_water(post[pol])[0]
    weights, channels = UNETS[method]
    model = UNet(channels)
    model.load_state_dict(torch.load(MODELS / f"{weights}.pt", map_location="cpu"))
    bands = [post["vv"], post["vh"]] + ([pre["vv"], pre["vh"]] if channels == 4 else [])
    return predict(model, np.stack(bands)) > 0.5


def pre_event(aoi: str, orbit: int, pass_id: str, grid, height) -> dict:
    path = PRE / f"{aoi}_{orbit}.npz"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **s1_grd.sigma0_pass(items(pass_id), grid, height))
    return dict(np.load(path))


def run(method: str) -> pd.DataFrame:
    cat = catalog()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for aoi in region_aois():
        grid = grid_of(aoi)
        anc = ancillary.build(aoi, grid)
        county = county_raster(grid)
        keep = (county > 0) & (anc["jrc_occurrence"] <= PERMANENT_WATER_OCCURRENCE)
        geom = gpd.GeoSeries([extent(SPEC[aoi])], crs=UTM).to_crs("EPSG:4326").iloc[0]
        pres = {}
        for p in covering(cat, geom, START, END):
            if p.orbit not in pres:
                cands = covering(cat, geom, PRE_EVENT_START, PRE_EVENT_END, **{"sat:relative_orbit": {"eq": p.orbit}})
                pres[p.orbit] = max(cands, key=lambda q: (round(q.cover, 2), q.date)) if cands else None
            if pres[p.orbit] is None:
                continue
            pre = pre_event(aoi, p.orbit, pres[p.orbit].id, grid, anc["ellipsoid_height"])
            post = s1_grd.sigma0_pass(p.items, grid, anc["ellipsoid_height"])
            valid = keep & np.isfinite(post["vv"]) & np.isfinite(pre["vv"])
            mask = flood_mask(method, post, pre) & valid
            with rasterio.open(OUT / f"{p.date:%Y%m%d}_o{p.orbit}_{aoi}.tif", "w", driver="GTiff", height=grid.height,
                               width=grid.width, count=1, dtype="uint8", crs=UTM, transform=grid.transform, nodata=255,
                               compress="deflate", tiled=True) as dst:
                dst.write(np.where(valid, mask, 255).astype("uint8"), 1)
            for code, name in COUNTY_NAMES.items():
                inside = county == code
                rows.append({"date": p.date.date().isoformat(), "orbit": p.orbit, "aoi": aoi, "county": name,
                             "observed_km2": (valid & inside).sum() * PX_KM2, "flooded_km2": (mask & inside).sum() * PX_KM2})
            print(aoi, p.date.date(), p.orbit, flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    method = ap.parse_args().method
    df = run(method)
    df["method"] = method
    df.to_csv(OUTPUTS / "tana_timeseries_tiles.csv", index=False)
    series = df.groupby(["orbit", "date", "county"])[["observed_km2", "flooded_km2"]].sum().reset_index()
    series.to_csv(OUTPUTS / "tana_timeseries.csv", index=False)
    print(series.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
