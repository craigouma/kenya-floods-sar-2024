"""Shared configuration for the Kenya 2024 Sentinel-1 flood mapping pipeline.

Every path, constant, and colour used by more than one stage is defined here so
that the notebook and the standalone scripts cannot drift apart.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REFERENCE = DATA / "reference"
AOI = DATA / "aoi"
ANCILLARY = DATA / "ancillary"
CACHE = DATA / "cache"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
PAPER = ROOT / "paper"

for _d in (REFERENCE, AOI, ANCILLARY, CACHE, MODELS, OUTPUTS, PAPER):
    _d.mkdir(parents=True, exist_ok=True)

ACTIVATION = "FL20240426KEN"
HDX_API = "https://data.humdata.org/api/3/action"

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Sentinel-1 search windows. The pre-event reference comes from the dry season
# and from the same relative orbit as the post-event scene (see pick_scenes.py).
MAX_DAY_GAP = 6
PRE_EVENT_START = "2024-01-01"
PRE_EVENT_END = "2024-02-29"

PIXEL_SIZE = 10

# Permanent water: JRC occurrence above this percentage is removed from every
# prediction and, where UNOSAT does not separate it, from the reference too.
PERMANENT_WATER_OCCURRENCE = 80
HAND_MAX_METRES = 20

BLOCK_SIZE_M = 1000
N_BOOTSTRAP = 1000
SEED = 20240426

KENYA_ADM_URL = (
    "https://data.humdata.org/dataset/2c0b7571-4bef-4347-9b81-b2174c13f9ef/"
    "resource/674ea496-5451-4312-bcab-8aa95fa3f36c/download/"
    "ken_admin_boundaries.geojson.zip"
)

# Portfolio palette (operating brief, section 1.3).
NAVY = "#1A3A5C"
BLUE = "#2563EB"
BLUE_LT = "#EFF6FF"
BG_ALT = "#F4F8FC"
BORDER = "#CBD8E6"
TEXT = "#1E2D3D"
MUTED = "#566B7E"
OMISSION = "#B45309"
COMMISSION = "#9CA3AF"

WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare",
    80: "Permanent water",
    90: "Herbaceous wetland",
}
