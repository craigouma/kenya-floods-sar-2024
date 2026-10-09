"""Run every flood mapping method on every AOI and write raw masks.

Masks are written before any post-processing (permanent water, HAND), which
evaluate.py applies identically to all methods.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

import classical
from config import MODELS, OUTPUTS, REFERENCE
from pick_scenes import SCENES
from prepare import stack
from unet import UNet, predict

MASKS = OUTPUTS / "masks"
MASKS.mkdir(parents=True, exist_ok=True)

UNETS = {"unet_post": ("post_hand_weak", 2), "unet_post_hand": ("post_hand", 2), "unet_pre": ("pre_hand", 4)}


def classical_masks(s: dict) -> tuple[dict, dict]:
    masks, thresholds = {}, {}
    for pol in ("vv", "vh"):
        post, pre = s[f"post_{pol}"], s[f"pre_{pol}"]
        masks[f"otsu_{pol}"], thresholds[f"otsu_{pol}"] = classical.otsu_water(post)
        masks[f"change_otsu_{pol}"], thresholds[f"change_otsu_{pol}"] = classical.change_otsu(post, pre)
        masks[f"change_fixed_{pol}"] = classical.change_fixed(post, pre)
        thresholds[f"change_fixed_{pol}"] = classical.FIXED_RATIO_DB
    return masks, thresholds


def unet_masks(s: dict) -> dict:
    masks = {}
    for name, (weights, channels) in UNETS.items():
        path = MODELS / f"{weights}.pt"
        if not path.exists():
            continue
        model = UNet(channels)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        bands = [s["post_vv"], s["post_vh"]] + ([s["pre_vv"], s["pre_vh"]] if channels == 4 else [])
        masks[name] = predict(model, np.stack(bands)) > 0.5
    return masks


def write(aoi: str, name: str, mask: np.ndarray, valid: np.ndarray) -> Path:
    with rasterio.open(REFERENCE / f"{aoi}_flood.tif") as ref:
        profile = ref.profile
    path = MASKS / f"{aoi}_{name}.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.where(valid, mask, 255).astype("uint8"), 1)
    return path


def run(aoi: str) -> list[dict]:
    s = stack(aoi)
    valid = np.isfinite(s["post_vv"]) & np.isfinite(s["pre_vv"])
    masks, thresholds = classical_masks(s)
    masks |= unet_masks(s)
    for name, mask in masks.items():
        write(aoi, name, mask, valid)
    return [{"aoi": aoi, "method": k, "threshold_db": thresholds.get(k)} for k in masks]


def main() -> None:
    aois = sys.argv[1:] or list(pd.read_csv(SCENES)["aoi"])
    rows = [r for aoi in aois for r in run(aoi)]
    df = pd.DataFrame(rows)
    path = OUTPUTS / "thresholds.csv"
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old["aoi"].isin(aois)], df])
    df.to_csv(path, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
