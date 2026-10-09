"""Download and catalogue every UNOSAT FL20240426KEN product on HDX.

UNOSAT attaches the same cumulative shapefile package for the activation to
every product page, growing as products are added, so only the most recently
modified archive is downloaded (to data/cache/). The figures UNOSAT reports in
each HDX dataset description (sensor, acquisition date, flooded area, affected
structures) are parsed into data/reference/catalogue.csv so every later
comparison cites them from one place.
"""

import re
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

from config import ACTIVATION, CACHE, HDX_API, REFERENCE

CATALOGUE = REFERENCE / "catalogue.csv"
ARCHIVE = CACHE / f"{ACTIVATION}_SHP.zip"

MONTHS = {"March": 3, "April": 4, "May": 5}


def search_products() -> list[dict]:
    r = requests.get(f"{HDX_API}/package_search", params={"q": ACTIVATION, "rows": 100}, timeout=60)
    r.raise_for_status()
    return [
        p
        for p in r.json()["result"]["results"]
        if ACTIVATION in (p.get("notes") or "") or any(ACTIVATION in res["url"] for res in p["resources"])
    ]


def parse_notes(notes: str) -> dict:
    sensor = re.search(r"(Sentinel-2|Pléiades|NOAA-VIIRS)", notes)
    date = re.search(r"acquired (?:on |between the )?(\d{1,2})(?:st|nd|rd|th)? (?:and the \d+\w* )?(?:of )?(\w+) 2024", notes)
    flooded = re.search(r"about (\d[\d,\.]*) km² of (land|cropland)\b", notes)
    analysed = re.search(r"analysed area of about (\d[\d,\.]*) km²", notes)
    structures = re.search(r"(\d[\d,]*) (?:affected structures|buildings are identified)", notes)
    return {
        "sensor": sensor.group(1) if sensor else None,
        "reference_date": f"2024-{MONTHS[date.group(2)]:02d}-{int(date.group(1)):02d}" if date else None,
        "reported_flooded_km2": float(flooded.group(1).replace(",", "")) if flooded else None,
        "reported_layer": flooded.group(2) if flooded else None,
        "reported_analysed_km2": float(analysed.group(1).replace(",", "")) if analysed else None,
        "reported_structures": int(structures.group(1).replace(",", "")) if structures else None,
    }


def archive_info(product: dict) -> dict:
    url = next(r["url"] for r in product["resources"] if r["url"].lower().endswith("_shp.zip"))
    head = requests.head(url, timeout=60)
    head.raise_for_status()
    return {
        "shp_url": url,
        "shp_bytes": int(head.headers["Content-Length"]),
        "shp_modified": parsedate_to_datetime(head.headers["Last-Modified"]).isoformat(),
    }


def download(url: str) -> None:
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(ARCHIVE, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def main() -> None:
    rows = []
    for p in search_products():
        rows.append(
            {
                "hdx_name": p["name"],
                "title": p["title"].strip(),
                "licence": p.get("license_title"),
                **parse_notes(p.get("notes") or ""),
                **archive_info(p),
                "notes": " ".join((p.get("notes") or "").split()),
            }
        )
    cat = pd.DataFrame(rows).sort_values("hdx_name")
    cat.to_csv(CATALOGUE, index=False)
    latest = cat.sort_values("shp_modified").iloc[-1]
    if not ARCHIVE.exists() or ARCHIVE.stat().st_size != latest["shp_bytes"]:
        download(latest["shp_url"])
    print(cat[["hdx_name", "sensor", "reference_date", "reported_flooded_km2", "reported_structures", "shp_bytes"]].to_string())
    print(f"cumulative archive: {latest['shp_url']}")


if __name__ == "__main__":
    main()
