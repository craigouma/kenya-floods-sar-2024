"""Ancillary layers resampled to each AOI's 10 m grid.

Copernicus DEM GLO-30, JRC Global Surface Water occurrence, and ESA WorldCover
2021 come from Planetary Computer; HAND is ASF's global GLO-30 HAND, derived
from the same DEM.
"""

import math

import numpy as np
import planetary_computer
import pyproj
import pystac_client
import rasterio
from rasterio.warp import Resampling, reproject

from config import CACHE, STAC_API
from s1_grd import Grid

HAND_URL = "https://glo-30-hand.s3.amazonaws.com/v1/2021/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_HAND.tif"

LAYERS = {
    "dem": ("cop-dem-glo-30", "data", Resampling.bilinear, {}),
    "jrc_occurrence": ("jrc-gsw", "occurrence", Resampling.nearest, {}),
    "worldcover": ("esa-worldcover", "map", Resampling.nearest, {"esa_worldcover:product_version": {"eq": "2.0.0"}}),
}


def _warp(hrefs: list[str], grid: Grid, resampling: Resampling, dtype: str = "float32") -> np.ndarray:
    out = np.full((grid.height, grid.width), np.nan if dtype == "float32" else 0, dtype=dtype)
    for href in hrefs:
        with rasterio.open(href) as src:
            dst = np.full_like(out, np.nan if dtype == "float32" else 0)
            reproject(rasterio.band(src, 1), dst, dst_transform=grid.transform, dst_crs=grid.crs,
                      resampling=resampling, dst_nodata=np.nan if dtype == "float32" else 0)
        fill = np.isnan(out) if dtype == "float32" else out == 0
        out[fill] = dst[fill]
    return out


def _bbox_4326(grid: Grid) -> list[float]:
    t = pyproj.Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    x0, y1 = grid.transform * (0, 0)
    x1, y0 = grid.transform * (grid.width, grid.height)
    lons, lats = t.transform([x0, x0, x1, x1], [y0, y1, y0, y1])
    return [min(lons), min(lats), max(lons), max(lats)]


def stac_layer(name: str, grid: Grid) -> np.ndarray:
    collection, asset, resampling, query = LAYERS[name]
    cat = pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)
    items = cat.search(collections=[collection], bbox=_bbox_4326(grid), query=query).item_collection()
    dtype = "float32" if name == "dem" else "uint8"
    return _warp([i.assets[asset].href for i in items], grid, resampling, dtype)


def hand(grid: Grid) -> np.ndarray:
    lon0, lat0, lon1, lat1 = _bbox_4326(grid)
    hrefs = []
    for lat in range(math.floor(lat0), math.floor(lat1) + 1):
        for lon in range(math.floor(lon0), math.floor(lon1) + 1):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            hrefs.append(HAND_URL.format(lat=ns, lon=ew))
    return _warp(hrefs, grid, Resampling.bilinear)


def ellipsoid_height(dem: np.ndarray, grid: Grid) -> np.ndarray:
    """Copernicus DEM heights are EGM2008 orthometric; GRD annotation heights are
    WGS84 ellipsoidal. The geoid undulation over Kenya is -15 to -30 m, which
    is two to four range pixels if left uncorrected."""
    pyproj.network.set_network_enabled(True)
    to_geo = pyproj.Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    to_ell = pyproj.Transformer.from_crs("EPSG:4326+3855", "EPSG:4979", always_xy=True)
    lon, lat = to_geo.transform(*grid.centres())
    _, _, h = to_ell.transform(lon, lat, np.nan_to_num(dem))
    return h.astype("float32")


def build(aoi: str, grid: Grid) -> dict[str, np.ndarray]:
    path = CACHE / "ancillary" / f"{aoi}.npz"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        return dict(np.load(path))
    layers = {name: stac_layer(name, grid) for name in LAYERS}
    layers["hand"] = hand(grid)
    layers["ellipsoid_height"] = ellipsoid_height(layers["dem"], grid)
    np.savez_compressed(path, **layers)
    return layers
