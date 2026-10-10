"""Classical methods on the Sen1Floods11 test and Bolivia splits.

Scored exactly as train_unet.py scores the U-Nets, so the rows of the
Sen1Floods11 table are comparable. Change detection needs a pre-event chip, so
it is scored on the paired chips only.
"""

import json

import numpy as np

import classical
from config import MODELS
from fetch_sen1floods11 import LOCAL
from pre_event_chips import OUT as PRE
from train_unet import iou_scores, label, read, split

METHODS = {
    "otsu_vv": lambda post, pre: classical.otsu_water(post[0])[0],
    "otsu_vh": lambda post, pre: classical.otsu_water(post[1])[0],
    "change_otsu_vv": lambda post, pre: classical.change_otsu(post[0], pre[0])[0],
    "change_otsu_vh": lambda post, pre: classical.change_otsu(post[1], pre[1])[0],
    "change_fixed_vv": lambda post, pre: classical.change_fixed(post[0], pre[0]),
    "change_fixed_vh": lambda post, pre: classical.change_fixed(post[1], pre[1]),
}


def main() -> None:
    hand = LOCAL / "HandLabeled"
    results = {}
    for name in ("test", "bolivia"):
        for suffix, pair in (("", None), ("_paired", PRE)):
            chips = split(LOCAL, name, pair)
            y = np.stack([label(hand / "LabelHand" / f"{c}_LabelHand.tif") for c in chips])
            perm = np.stack([read(hand / "JRCWaterHand" / f"{c}_JRCWaterHand.tif")[0] == 1 for c in chips])
            post = [read(hand / "S1Hand" / f"{c}_S1Hand.tif") for c in chips]
            pre = [read(PRE / f"{c}_S1PreHand.tif") for c in chips] if pair else None
            for method, fn in METHODS.items():
                if method.startswith("change") and pre is None:
                    continue
                pred = np.stack([fn(p, pre[i] if pre else None) for i, p in enumerate(post)]).astype("uint8")
                results.setdefault(method, {})[name + suffix] = {
                    "all_water": iou_scores(pred, y), "flood_water": iou_scores(pred, y, ~perm)}
    (MODELS / "sen1floods11_classical.json").write_text(json.dumps(results, indent=1))
    for method, r in results.items():
        print(method, {k: round(v["flood_water"]["mean_chip_iou"], 3) for k, v in r.items()})


if __name__ == "__main__":
    main()
