"""Download Sen1Floods11 v1.1 and record its flood events.

Fetches the Sentinel-1 chips, hand labels, S2-derived weak labels, JRC water
chips, and split lists from the public `sen1floods11` bucket (anonymous
access), skipping files already cached. Writes data/sen1floods11_events.csv
with each event's date, orbit, and whether it lies in Africa or East Africa,
which bounds how far the Kenya evaluation is out of region.
"""

import json

import gcsfs
import pandas as pd

from config import CACHE, DATA

BUCKET = "sen1floods11/v1.1"
LOCAL = CACHE / "sen1floods11"
EVENTS = DATA / "sen1floods11_events.csv"

FOLDERS = [
    "data/flood_events/HandLabeled/S1Hand",
    "data/flood_events/HandLabeled/LabelHand",
    "data/flood_events/HandLabeled/JRCWaterHand",
    "data/flood_events/WeaklyLabeled/S1Weak",
    "data/flood_events/WeaklyLabeled/S2IndexLabelWeak",
    "splits/flood_handlabeled",
]

AFRICA = {"GHA", "NGA", "SOM"}
EAST_AFRICA = {"SOM"}
CHIP_PREFIX = {"Cambodia": "Mekong"}


def fetch(fs: gcsfs.GCSFileSystem) -> None:
    fs.get(f"{BUCKET}/Sen1Floods11_Metadata.geojson", str(LOCAL / "Sen1Floods11_Metadata.geojson"))
    for folder in FOLDERS:
        for remote in fs.ls(f"{BUCKET}/{folder}"):
            target = LOCAL / remote.removeprefix(f"{BUCKET}/").replace("data/flood_events/", "")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                fs.get(remote, str(target))


def events() -> pd.DataFrame:
    meta = json.loads((LOCAL / "Sen1Floods11_Metadata.geojson").read_text())
    rows = {f["properties"]["ID"]: f["properties"] for f in meta["features"]}
    hand = sorted((LOCAL / "HandLabeled" / "S1Hand").glob("*.tif"))
    hand_counts = pd.Series([p.name.split("_")[0] for p in hand]).value_counts()
    df = pd.DataFrame(rows.values())
    df["chip_prefix"] = df["location"].replace(CHIP_PREFIX)
    df["hand_chips"] = df["chip_prefix"].map(hand_counts).fillna(0).astype(int)
    df["africa"] = df["ISO_CC"].isin(AFRICA)
    df["east_africa"] = df["ISO_CC"].isin(EAST_AFRICA)
    cols = ["ID", "location", "chip_prefix", "ISO_CC", "s1_date", "s2_date", "orbit", "rel_orbit_num",
            "VH_thresh", "hand_chips", "africa", "east_africa"]
    return df[cols].sort_values("ID")


def main() -> None:
    fetch(gcsfs.GCSFileSystem(token="anon"))
    df = events()
    df.to_csv(EVENTS, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
