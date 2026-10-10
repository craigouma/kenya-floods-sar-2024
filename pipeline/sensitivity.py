"""Sensitivity of agreement to the SAR-to-reference day gap.

Every Sentinel-1 pass within WINDOW_DAYS of an AOI's reference date is run
through the same methods and scored against the same reference, each paired
with its own orbit's dry-season scene. Floods rise and recede, so a pass three
days before the reference image and one three days after are not equivalent.
"""

import numpy as np
import pandas as pd

from config import OUTPUTS, PERMANENT_WATER_OCCURRENCE, PRE_EVENT_END, PRE_EVENT_START
from methods import classical_masks, unet_masks
from metrics import counts, scores
from pick_scenes import catalog, covering
from prepare import stack
from reference import SPEC, load_aois

WINDOW_DAYS = 24


def passes(cat, geom, ref_date: pd.Timestamp) -> list[tuple[str, str, int]]:
    post = covering(cat, geom, f"{ref_date - pd.Timedelta(days=WINDOW_DAYS):%Y-%m-%d}",
                    f"{ref_date + pd.Timedelta(days=WINDOW_DAYS):%Y-%m-%d}")
    out = []
    for p in post:
        pre = covering(cat, geom, PRE_EVENT_START, PRE_EVENT_END, **{"sat:relative_orbit": {"eq": p.orbit}})
        if pre:
            out.append((p.id, max(pre, key=lambda q: q.date).id, p.orbit))
    return out


def main(force: bool = False) -> None:
    out = OUTPUTS / "day_gap_sensitivity.csv"
    if out.exists() and not force:
        print(f"using {out.name}; pass force=True to recompute")
        return
    cat = catalog()
    rows = []
    aois = load_aois()
    for a in aois[~aois["aoi"].str.startswith("tana_")].to_crs("EPSG:4326").itertuples():
        ref_date = pd.Timestamp(a.reference_date)
        for post_id, pre_id, orbit in passes(cat, a.geometry, ref_date):
            s = stack(a.aoi, post_id, pre_id)
            valid = (s["reference"] != 255) & np.isfinite(s["post_vv"]) & np.isfinite(s["pre_vv"])
            valid &= s["jrc_occurrence"] <= PERMANENT_WATER_OCCURRENCE
            masks, _ = classical_masks(s)
            masks |= unet_masks(s)
            gap = (pd.Timestamp(post_id[17:25]) - ref_date).days
            for method, m in masks.items():
                c = counts(m, s["reference"] == 1, valid)
                rows.append({"aoi": a.aoi, "group": SPEC[a.aoi].group, "post_id": post_id, "orbit": orbit,
                             "day_gap": gap, "method": method, **scores(c),
                             "pred_km2": c[[0, 1]].sum() / 1e4, "ref_km2": c[[0, 2]].sum() / 1e4})
            print(a.aoi, gap, orbit, flush=True)
    df = pd.DataFrame(rows).sort_values(["aoi", "method", "day_gap"])
    df.to_csv(out, index=False)
    best = df[df["method"].isin(["change_fixed_vv", "change_otsu_vv", "otsu_vh"])]
    print(best.pivot_table(index=["aoi", "day_gap"], columns="method", values="iou").round(3).to_string())


if __name__ == "__main__":
    main()
