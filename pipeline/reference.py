"""UNOSAT reference layers: AOI definitions and rasterisation to the SAR grid.

All four urban HDX products (Nairobi; Nairobi and Kiambu; Nairobi and
Machakos; Kiambu) are presentations of a single Pléiades analysis of 1 May
2024 with one flood layer and one analysis extent; their structure counts sum
to the 13,111 points in the shared structure layer. They are therefore one
AOI here, reported by county.

The Sentinel-2 product of 29 April for Garissa has no flood layer of its own in
the archive. Its flood water is the Tana and Galana rivers layer from the same
image, clipped to the Garissa analysis extent.
"""

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from config import AOI, PIXEL_SIZE, REFERENCE
from fetch_reference import ARCHIVE

UTM = "EPSG:32737"
NODATA = 255


@dataclass(frozen=True)
class Spec:
    aoi: str
    group: str
    sensor: str
    reference_date: str
    extent: str
    flood: str
    cloud: str | None
    structures: str | None
    hdx: tuple[str, ...]


SPECS = [
    Spec("garissa_s2", "riverine", "Sentinel-2", "2024-04-29",
         "S2_20240429_AnalysisExtent_Garissa",
         "S2_20240429_FloodExtent_AlongTanaRiver_GalanaRiver",
         "S2_20240429_CloudObstruction_AlongTanaRiver_GalanaRiver",
         "S2_20240429_PotentiallyAffectedStructure_Garissa",
         ("water-extent-in-garissa-city-and-mororo-village-kenya-as-of-29-april-2024",)),
    Spec("garissa_pl", "riverine", "Pléiades", "2024-05-05",
         "PL_20240505_AnalysisExtent_Garissa",
         "PL_20240505_FloodExtent_Garissa",
         None,
         "PL_20240505_AffectedStructure_Garissa",
         ("satellite-detected-water-extent-in-garissa-city-and-mororo-village-kenya-as-of-5-may-2024",)),
    Spec("nairobi_pl", "urban", "Pléiades", "2024-05-01",
         "PL_20240501_AnalysisExtent_Nairobi_Kiambu",
         "PL_20240501_FloodExtent_Nairobi_Kiambu",
         "PL_20240501_CloudObstruction_Nairobi_Kiambu",
         "PL_20240501_AffectedStructureNairobi_Kiambu",
         ("flood-impact-analysis-in-nairobi-kenya-as-of-1-may-2024",
          "flood-impact-analysis-in-nairobi-and-kiambu-county-kenya-as-of-1-may-2024",
          "flood-impact-analysis-in-nairobi-and-machakos-county-kenya-as-of-1-may-2024",
          "flood-impact-analysis-in-kiambu-county-kenya-as-of-1-may-2024")),
]
SPEC = {s.aoi: s for s in SPECS}


def layer(name: str) -> gpd.GeoDataFrame:
    return gpd.read_file(f"zip://{ARCHIVE}!FL20240426KEN_SHP/{name}.shp")


def domain(spec: Spec):
    """Analysed extent minus areas UNOSAT marked as obscured by cloud."""
    extent = layer(spec.extent).to_crs(UTM).union_all()
    if spec.cloud:
        extent = extent.difference(layer(spec.cloud).to_crs(UTM).clip(extent.bounds).union_all())
    return extent


def load_aois() -> gpd.GeoDataFrame:
    rows = [{"aoi": s.aoi, "group": s.group, "sensor": s.sensor, "reference_date": s.reference_date,
             "geometry": layer(s.extent).to_crs(UTM).union_all()} for s in SPECS]
    return gpd.GeoDataFrame(rows, crs=UTM)


def grid_for(geom) -> tuple[rasterio.Affine, int, int]:
    x0, y0, x1, y1 = geom.bounds
    x0, y0 = np.floor(x0 / PIXEL_SIZE) * PIXEL_SIZE, np.floor(y0 / PIXEL_SIZE) * PIXEL_SIZE
    x1, y1 = np.ceil(x1 / PIXEL_SIZE) * PIXEL_SIZE, np.ceil(y1 / PIXEL_SIZE) * PIXEL_SIZE
    width, height = int((x1 - x0) / PIXEL_SIZE), int((y1 - y0) / PIXEL_SIZE)
    return rasterio.Affine(PIXEL_SIZE, 0, x0, 0, -PIXEL_SIZE, y1), width, height


def rasterise(spec: Spec) -> None:
    """Write the reference flood raster: 1 flood, 0 dry, 255 outside the domain.

    Polygons are reprojected from EPSG:4326 to UTM 37S before burning, and a
    pixel counts as flooded when its centre falls inside a flood polygon.
    """
    dom = domain(spec)
    transform, width, height = grid_for(layer(spec.extent).to_crs(UTM).union_all())
    shape = (height, width)
    inside = rasterize([dom], out_shape=shape, transform=transform, fill=0, default_value=1, dtype="uint8")
    flood_geoms = layer(spec.flood).to_crs(UTM).clip(dom.bounds).explode(index_parts=False).geometry
    flood = rasterize(((g, 1) for g in flood_geoms if not g.is_empty), out_shape=shape,
                      transform=transform, fill=0, dtype="uint8")
    out = np.where(inside == 1, flood, NODATA).astype("uint8")
    profile = dict(driver="GTiff", height=height, width=width, count=1, dtype="uint8", crs=UTM,
                   transform=transform, nodata=NODATA, compress="deflate", predictor=2, tiled=True)
    with rasterio.open(REFERENCE / f"{spec.aoi}_flood.tif", "w", **profile) as dst:
        dst.write(out, 1)


def summary() -> pd.DataFrame:
    rows = []
    for s in SPECS:
        with rasterio.open(REFERENCE / f"{s.aoi}_flood.tif") as src:
            a = src.read(1)
        px_km2 = PIXEL_SIZE**2 / 1e6
        rows.append({"aoi": s.aoi, "group": s.group, "sensor": s.sensor, "reference_date": s.reference_date,
                     "domain_km2": round((a != NODATA).sum() * px_km2, 2),
                     "flood_km2": round((a == 1).sum() * px_km2, 2),
                     "excluded_cloud_km2": round(layer(s.extent).to_crs(UTM).area.sum() / 1e6 - (a != NODATA).sum() * px_km2, 2),
                     "hdx_products": ";".join(s.hdx)})
    return pd.DataFrame(rows)


def main() -> None:
    for s in SPECS:
        rasterise(s)
    load_aois().to_file(AOI / "aois.gpkg", driver="GPKG")
    df = summary()
    df.to_csv(REFERENCE / "reference_summary.csv", index=False)
    print(df.drop(columns="hdx_products").to_string(index=False))


if __name__ == "__main__":
    main()
