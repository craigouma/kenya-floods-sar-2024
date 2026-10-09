"""Tana and Galana river corridor tiles for evaluation and the time series.

The corridor is defined from JRC surface water, not from the reference: land
within CORRIDOR_M of any 300 m cell where JRC water occurrence reaches
RIVER_OCCURRENCE at 30 m. It is cut into TILE_M square UTM tiles; tiles keep
the part inside UNOSAT's Tana and Galana analysis extent and outside the
Garissa extent, which is evaluated on its own.
"""

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.features import shapes
from rasterio.warp import Resampling
from shapely.geometry import box, shape
from shapely.ops import unary_union

from ancillary import _warp
from boundaries import counties
from config import STAC_API
from reference import SPECS, TANA_TILES, UTM, layer_utm
from s1_grd import Grid

RIVER_OCCURRENCE = 50
COARSE_M = 300
CORRIDOR_M = 5000
TILE_M = 20000
MIN_TILE_KM2 = 20
TANA_EXTENT = "S2_20240429_AnalysisExtent_AlongTanaRiver_GalanaRiver"


def river_cells(region) -> gpd.GeoSeries:
    x0, y0, x1, y1 = region.bounds
    w, h = int((x1 - x0) // COARSE_M) + 1, int((y1 - y0) // COARSE_M) + 1
    grid = Grid(UTM, rasterio.Affine(COARSE_M, 0, x0, 0, -COARSE_M, y1), w, h)
    cat = pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)
    lon0, lat0, lon1, lat1 = gpd.GeoSeries([region], crs=UTM).to_crs("EPSG:4326").total_bounds
    items = cat.search(collections=["jrc-gsw"], bbox=[lon0, lat0, lon1, lat1]).item_collection()
    occ = _warp([i.assets["occurrence"].href for i in items], grid, Resampling.max, "uint8")
    water = ((occ >= RIVER_OCCURRENCE) & (occ <= 100)).astype("uint8")
    polys = [shape(g) for g, v in shapes(water, mask=water == 1, transform=grid.transform)]
    return gpd.GeoSeries(polys, crs=UTM)


def corridor():
    region = layer_utm(TANA_EXTENT).union_all()
    land = counties().to_crs(UTM).union_all()
    rivers = river_cells(region)
    return unary_union(rivers.buffer(CORRIDOR_M)).intersection(region).intersection(land)


def main() -> None:
    zone = corridor().difference(layer_utm(SPECS[0].extent).union_all())
    x0, y0, x1, y1 = zone.bounds
    tiles = []
    for x in np.arange(np.floor(x0 / TILE_M) * TILE_M, x1, TILE_M):
        for y in np.arange(np.floor(y0 / TILE_M) * TILE_M, y1, TILE_M):
            part = zone.intersection(box(x, y, x + TILE_M, y + TILE_M))
            if part.area / 1e6 >= MIN_TILE_KM2:
                tiles.append({"aoi": f"tana_{int(x / 1000)}_{int(y / 1000)}", "geometry": part})
    gdf = gpd.GeoDataFrame(tiles, crs=UTM)
    gdf.to_file(TANA_TILES, driver="GPKG")
    print(f"{len(gdf)} tiles, {gdf.area.sum() / 1e6:,.0f} km2 of corridor")


if __name__ == "__main__":
    main()
