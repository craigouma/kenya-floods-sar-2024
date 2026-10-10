"""Building exposure: Open Buildings v3 centroids inside each flood mask.

UNOSAT counted affected structures on its own very high resolution imagery,
not on Open Buildings, so three numbers are reported per AOI: buildings inside
each method's mask, buildings inside UNOSAT's own flood extent (isolating the
building dataset from the flood extent), and UNOSAT's structure count.
"""

import gzip

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from config import CACHE, OUTPUTS, PERMANENT_WATER_OCCURRENCE
from evaluate import load_mask, methods_for
from prepare import stack
from reference import SPEC, UTM, domain, layer

TILE = CACHE / "buildings" / "183_buildings.csv.gz"
AOIS = {a: s for a, s in SPEC.items() if s.structures}
MIN_CONFIDENCE = 0.70
SUBSET = CACHE / "buildings" / "aoi_buildings.csv"


def buildings() -> pd.DataFrame:
    if SUBSET.exists():
        return pd.read_csv(SUBSET)
    bounds = pd.DataFrame([layer(s.extent).total_bounds for s in AOIS.values()], columns=["x0", "y0", "x1", "y1"])
    x0, y0, x1, y1 = bounds["x0"].min(), bounds["y0"].min(), bounds["x1"].max(), bounds["y1"].max()
    parts = []
    with gzip.open(TILE, "rt") as f:
        for chunk in pd.read_csv(f, chunksize=2_000_000, usecols=["latitude", "longitude", "area_in_meters", "confidence"]):
            parts.append(chunk[chunk["longitude"].between(x0, x1) & chunk["latitude"].between(y0, y1)])
    df = pd.concat(parts)
    df.to_csv(SUBSET, index=False)
    return df


def pixels(df: pd.DataFrame, transform) -> tuple[np.ndarray, np.ndarray]:
    x, y = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    col, row = ~transform * (x, y)
    return np.floor(row).astype(int), np.floor(col).astype(int)


def main() -> None:
    b = buildings()
    b = b[b["confidence"] >= MIN_CONFIDENCE]
    rows = []
    for aoi, spec in AOIS.items():
        s = stack(aoi)
        with rasterio.open(OUTPUTS / "masks" / f"{aoi}_{methods_for(aoi)[0]}.tif") as src:
            transform, shape = src.transform, src.shape
        r, c = pixels(b, transform)
        inside = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        r, c = r[inside], c[inside]
        valid = (s["reference"] != 255) & (s["jrc_occurrence"] <= PERMANENT_WATER_OCCURRENCE)
        in_domain = valid[r, c]
        unosat_points = layer(spec.structures).to_crs(UTM)
        unosat_in_domain = int(unosat_points.within(domain(spec)).sum())
        base = {"aoi": aoi, "group": spec.group, "buildings_in_domain": int(in_domain.sum()),
                "unosat_structures_in_domain": unosat_in_domain, "unosat_structures_layer": len(unosat_points)}
        rows.append({**base, "method": "unosat_extent", "exposed": int(((s["reference"] == 1)[r, c] & in_domain).sum())})
        for method in methods_for(aoi):
            m = load_mask(aoi, method) == 1
            rows.append({**base, "method": method, "exposed": int((m[r, c] & in_domain).sum())})
    df = pd.DataFrame(rows)
    df["ratio_to_unosat"] = df["exposed"] / df["unosat_structures_in_domain"]
    df.to_csv(OUTPUTS / "exposure.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
