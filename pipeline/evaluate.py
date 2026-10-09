"""Score every method mask against the UNOSAT reference.

Post-processing, identical for all methods: pixels with JRC occurrence above
PERMANENT_WATER_OCCURRENCE are removed from the prediction and from the
evaluation domain (UNOSAT's layers record flood water, so permanent water is
excluded on both sides). The HAND variant additionally sets pixels more than
HAND_MAX_METRES above the nearest drainage to dry.

Outputs per AOI, per group, and per WorldCover class, each with a 95% interval
from resampling 1 km blocks.
"""

import numpy as np
import pandas as pd
import rasterio

from config import (BLOCK_SIZE_M, HAND_MAX_METRES, OUTPUTS, PERMANENT_WATER_OCCURRENCE, PIXEL_SIZE,
                    WORLDCOVER_CLASSES)
from metrics import block_counts, bootstrap, counts, scores
from methods import MASKS
from prepare import stack
from reference import SPEC

BLOCK = BLOCK_SIZE_M // PIXEL_SIZE
PX_KM2 = PIXEL_SIZE**2 / 1e6


def methods_for(aoi: str) -> list[str]:
    return sorted(p.stem.removeprefix(f"{aoi}_") for p in MASKS.glob(f"{aoi}_*.tif"))


def load_mask(aoi: str, method: str) -> np.ndarray:
    with rasterio.open(MASKS / f"{aoi}_{method}.tif") as src:
        return src.read(1)


def prepared(aoi: str, method: str, hand: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    s = stack(aoi)
    raw = load_mask(aoi, method)
    permanent = s["jrc_occurrence"] > PERMANENT_WATER_OCCURRENCE
    valid = (s["reference"] != 255) & (raw != 255) & ~permanent
    pred = raw == 1
    if hand:
        pred &= ~(s["hand"] > HAND_MAX_METRES)
    return pred, s["reference"] == 1, valid, s


def evaluate_aoi(aoi: str) -> tuple[list[dict], list[dict], dict]:
    rows, lc_rows, blocks = [], [], {}
    for method in methods_for(aoi):
        for hand in (False, True):
            pred, ref, valid, s = prepared(aoi, method, hand)
            c = counts(pred, ref, valid)
            b = block_counts(pred, ref, valid, BLOCK)
            blocks[(method, hand)] = b
            ci = bootstrap(b)
            rows.append({"aoi": aoi, "group": SPEC[aoi].group, "method": method, "hand_mask": hand,
                         **{k: float(v) for k, v in scores(c).items()},
                         **{f"{k}_lo": v[0] for k, v in ci.items()}, **{f"{k}_hi": v[1] for k, v in ci.items()},
                         "tp": c[0], "fp": c[1], "fn": c[2], "tn": c[3],
                         "pred_km2": (pred & valid).sum() * PX_KM2, "ref_km2": (ref & valid).sum() * PX_KM2,
                         "domain_km2": valid.sum() * PX_KM2, "n_blocks": len(b)})
            for code, name in WORLDCOVER_CLASSES.items():
                v = valid & (s["worldcover"] == code)
                if v.sum() * PX_KM2 < 0.5:
                    continue
                lc = counts(pred, ref, v)
                lc_rows.append({"aoi": aoi, "group": SPEC[aoi].group, "method": method, "hand_mask": hand,
                                "worldcover": name, "domain_km2": v.sum() * PX_KM2, "ref_km2": lc[[0, 2]].sum() * PX_KM2,
                                "tp": lc[0], "fp": lc[1], "fn": lc[2], "tn": lc[3],
                                **{k: float(x) for k, x in scores(lc).items()}})
    return rows, lc_rows, blocks


def pooled(rows: pd.DataFrame, blocks: dict) -> pd.DataFrame:
    out = []
    for (group, method, hand), g in rows.groupby(["group", "method", "hand_mask"]):
        c = g[["tp", "fp", "fn", "tn"]].sum().to_numpy()
        b = np.concatenate([blocks[(a, method, hand)] for a in g["aoi"]])
        ci = bootstrap(b)
        out.append({"group": group, "method": method, "hand_mask": hand, "aois": ";".join(g["aoi"]),
                    **{k: float(v) for k, v in scores(c).items()},
                    **{f"{k}_lo": v[0] for k, v in ci.items()}, **{f"{k}_hi": v[1] for k, v in ci.items()},
                    "tp": c[0], "fp": c[1], "fn": c[2], "tn": c[3]})
    return pd.DataFrame(out)


def main() -> None:
    rows, lc_rows, blocks = [], [], {}
    for aoi in SPEC:
        r, lc, b = evaluate_aoi(aoi)
        rows += r
        lc_rows += lc
        blocks |= {(aoi, *k): v for k, v in b.items()}
    per_aoi = pd.DataFrame(rows)
    per_aoi.to_csv(OUTPUTS / "metrics_by_aoi.csv", index=False)
    groups = pooled(per_aoi, blocks)
    groups.to_csv(OUTPUTS / "metrics_by_group.csv", index=False)
    lc = pd.DataFrame(lc_rows)
    lc_group = lc.groupby(["group", "method", "hand_mask", "worldcover"])[["domain_km2", "ref_km2", "tp", "fp", "fn", "tn"]].sum()
    lc_group = lc_group.join(pd.DataFrame(scores(lc_group[["tp", "fp", "fn", "tn"]].to_numpy()), index=lc_group.index))
    lc_group.reset_index().to_csv(OUTPUTS / "metrics_by_landcover.csv", index=False)
    cols = ["group", "method", "hand_mask", "iou", "iou_lo", "iou_hi", "f1", "precision", "recall"]
    print(groups[cols].round(3).sort_values(["group", "hand_mask", "iou"], ascending=[True, True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
