"""Choose the post-event and pre-event Sentinel-1 scene for every AOI.

Post-event: the GRD acquisition closest in time to the UNOSAT reference image,
within MAX_DAY_GAP days, that covers the whole AOI. AOIs without one keep their
nearest pass but are flagged for the sensitivity table only.

Pre-event: the latest January to February 2024 acquisition from the same
relative orbit, so that look geometry matches and the late-2023 short-rains
flooding has had the longest time to recede.
"""

import pandas as pd
import planetary_computer
import pystac_client
from shapely.geometry import shape

from config import AOI, MAX_DAY_GAP, PRE_EVENT_END, PRE_EVENT_START, STAC_API
from reference import load_aois

SCENES = AOI / "scenes.csv"
SEARCH_DAYS = 30


def catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)


def covering(cat, geom, start: str, end: str, **query) -> list:
    items = cat.search(collections=["sentinel-1-grd"], intersects=geom, datetime=f"{start}/{end}",
                       query={"sar:instrument_mode": {"eq": "IW"}, **query}).item_collection()
    return [i for i in items if shape(i.geometry).contains(geom) and {"vv", "vh"} <= set(i.assets)]


def pick(cat, geom, ref_date: pd.Timestamp) -> dict:
    window = (ref_date - pd.Timedelta(days=SEARCH_DAYS), ref_date + pd.Timedelta(days=SEARCH_DAYS))
    items = covering(cat, geom, window[0].strftime("%Y-%m-%d"), window[1].strftime("%Y-%m-%d"))
    if not items:
        raise ValueError(f"no Sentinel-1 GRD scene covers the AOI within {SEARCH_DAYS} days")
    gap = lambda i: (pd.Timestamp(i.datetime).tz_localize(None).normalize() - ref_date).days
    post = min(items, key=lambda i: (abs(gap(i)), gap(i)))
    orbit = post.properties["sat:relative_orbit"]
    pre_items = covering(cat, geom, PRE_EVENT_START, PRE_EVENT_END, **{"sat:relative_orbit": {"eq": orbit}})
    if not pre_items:
        raise ValueError(f"no dry-season scene on relative orbit {orbit}")
    pre = max(pre_items, key=lambda i: i.datetime)
    return {
        "post_id": post.id,
        "post_date": post.datetime.date().isoformat(),
        "day_gap": gap(post),
        "headline": abs(gap(post)) <= MAX_DAY_GAP,
        "pre_id": pre.id,
        "pre_date": pre.datetime.date().isoformat(),
        "relative_orbit": orbit,
        "orbit_state": post.properties["sat:orbit_state"],
        "n_candidates_within_gap": sum(abs(gap(i)) <= MAX_DAY_GAP for i in items),
    }


def main() -> None:
    cat = catalog()
    aois = load_aois().to_crs("EPSG:4326")
    rows = [{"aoi": a.aoi, "reference_date": a.reference_date, **pick(cat, a.geometry, pd.Timestamp(a.reference_date))}
            for a in aois.itertuples()]
    df = pd.DataFrame(rows)
    df.to_csv(SCENES, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
