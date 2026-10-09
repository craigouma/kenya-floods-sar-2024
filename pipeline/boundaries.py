"""Kenya county boundaries (HDX COD-AB, ADM1), as used in Paper 6."""

import zipfile

import geopandas as gpd
import requests

from config import CACHE, KENYA_ADM_URL

ZIP = CACHE / "ken_admin_boundaries.geojson.zip"
ADM1 = CACHE / "ken_admin1.geojson"


def counties() -> gpd.GeoDataFrame:
    if not ADM1.exists():
        if not ZIP.exists():
            r = requests.get(KENYA_ADM_URL, timeout=300)
            r.raise_for_status()
            ZIP.write_bytes(r.content)
        with zipfile.ZipFile(ZIP) as z:
            z.extract("ken_admin1.geojson", CACHE)
    gdf = gpd.read_file(ADM1)[["adm1_name", "adm1_pcode", "geometry"]]
    return gdf.rename(columns={"adm1_name": "county", "adm1_pcode": "pcode"}).to_crs("EPSG:4326")
