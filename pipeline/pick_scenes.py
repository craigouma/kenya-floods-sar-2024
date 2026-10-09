"""Choose the post-event and pre-event Sentinel-1 pass for every AOI.

A pass is every GRD slice of one datatake that touches the AOI, since an AOI
can straddle two consecutive slices. Pass ids are the slice ids joined with
';'. A pass must image at least MIN_COVER of the AOI; corridor tiles that
straddle two tracks keep the larger share, and unimaged pixels drop out of the
evaluation as no-data.

Post-event: the pass closest in time to the UNOSAT reference image. Passes
more than MAX_DAY_GAP days away are kept but flagged, and leave the AOI out of
the headline numbers.

Pre-event: the latest January to February 2024 pass from the same relative
orbit, so that look geometry matches and the late-2023 short-rains flooding
has had the longest time to recede.
"""

from dataclasses import dataclass

import pandas as pd
import planetary_computer
import pystac_client
from shapely.geometry import shape
from shapely.ops import unary_union

from config import AOI, MAX_DAY_GAP, PRE_EVENT_END, PRE_EVENT_START, STAC_API
from reference import load_aois

SCENES = AOI / "scenes.csv"
SEARCH_DAYS = 30
MIN_COVER = 0.5


@dataclass
class Pass:
    items: list
    cover: float

    @property
    def id(self) -> str:
        return ";".join(sorted(i.id for i in self.items))

    @property
    def date(self) -> pd.Timestamp:
        return pd.Timestamp(self.items[0].datetime).tz_localize(None).normalize()

    @property
    def orbit(self) -> int:
        return self.items[0].properties["sat:relative_orbit"]


def catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)


def covering(cat, geom, start: str, end: str, **query) -> list[Pass]:
    items = cat.search(collections=["sentinel-1-grd"], intersects=geom, datetime=f"{start}/{end}",
                       query={"sar:instrument_mode": {"eq": "IW"}, **query}).item_collection()
    groups = {}
    for i in items:
        if {"vv", "vh"} <= set(i.assets):
            groups.setdefault((i.datetime.date(), i.properties["sat:relative_orbit"]), []).append(i)
    passes = [Pass(g, unary_union([shape(i.geometry) for i in g]).intersection(geom).area / geom.area)
              for g in groups.values()]
    return [p for p in passes if p.cover >= MIN_COVER]


def pick(cat, geom, ref_date: pd.Timestamp) -> dict:
    window = (ref_date - pd.Timedelta(days=SEARCH_DAYS), ref_date + pd.Timedelta(days=SEARCH_DAYS))
    passes = covering(cat, geom, f"{window[0]:%Y-%m-%d}", f"{window[1]:%Y-%m-%d}")
    if not passes:
        raise ValueError(f"no Sentinel-1 pass covers the AOI within {SEARCH_DAYS} days")
    gap = lambda p: (p.date - ref_date).days
    post = min(passes, key=lambda p: (abs(gap(p)), -p.cover, gap(p)))
    pre_passes = covering(cat, geom, PRE_EVENT_START, PRE_EVENT_END, **{"sat:relative_orbit": {"eq": post.orbit}})
    if not pre_passes:
        raise ValueError(f"no dry-season pass on relative orbit {post.orbit}")
    pre = max(pre_passes, key=lambda p: (p.cover, p.date))
    return {
        "post_id": post.id,
        "post_date": post.date.date().isoformat(),
        "day_gap": gap(post),
        "headline": abs(gap(post)) <= MAX_DAY_GAP,
        "pre_id": pre.id,
        "pre_date": pre.date.date().isoformat(),
        "post_cover": round(post.cover, 3),
        "pre_cover": round(pre.cover, 3),
        "relative_orbit": post.orbit,
        "orbit_state": post.items[0].properties["sat:orbit_state"],
        "n_passes_within_gap": sum(abs(gap(p)) <= MAX_DAY_GAP for p in passes),
    }


def main() -> None:
    cat = catalog()
    aois = load_aois().to_crs("EPSG:4326")
    rows = [{"aoi": a.aoi, "reference_date": a.reference_date, **pick(cat, a.geometry, pd.Timestamp(a.reference_date))}
            for a in aois.itertuples()]
    df = pd.DataFrame(rows)
    df.to_csv(SCENES, index=False)
    print(df.drop(columns=["post_id", "pre_id"]).to_string(index=False))


if __name__ == "__main__":
    main()
