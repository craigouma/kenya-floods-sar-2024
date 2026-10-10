# Sentinel-1 flood mapping of the 2024 Kenya long-rains floods

How well do Sentinel-1 flood mapping methods trained or tuned elsewhere work on
Kenyan landscapes? This repository runs classical and deep learning methods
over the March to May 2024 floods and scores them against UNOSAT's rapid
mapping of the event (activation FL20240426KEN). Riverine and urban settings
are reported separately and broken down by land cover. The repository then
maps the lower Tana through April to June 2024 with the best riverine method.

**Paper:** [`paper/kenya_floods_sar.pdf`](paper/kenya_floods_sar.pdf) ·
**Landing page:** https://craigouma.github.io/kenya-floods-sar-2024/ ·
**Decisions log:** [`DECISIONS.md`](DECISIONS.md)

Craig Carlos Ouma, Independent Researcher.

## Findings

| Method | Riverine IoU (95% CI) | Urban IoU (95% CI) |
|---|---|---|
| Otsu threshold, VH | 0.116 (0.109 to 0.125) | 0.192 (0.161 to 0.225) |
| Change detection, -3 dB, VV | 0.160 (0.150 to 0.172) | 0.151 (0.115 to 0.189) |
| U-Net, Sen1Floods11 hand and weak labels | **0.347 (0.331 to 0.364)** | 0.221 (0.171 to 0.273) |
| U-Net, pre- and post-event, hand labels | 0.277 (0.262 to 0.293) | 0.228 (0.175 to 0.282) |

- The U-Net trained only on Sen1Floods11 (test all-water IoU 0.424, against
  the published 0.408) roughly doubles the agreement of the best classical
  method along the Tana, and its flooded area is within about 1% of UNOSAT's
  on two references.
- In the Nairobi, Kiambu and Machakos analysis no method is clearly better
  than another. Built-up land is where all nine methods fail (flood-class F1
  below 0.11).
- The radar-to-optical day gap matters as much as the method: at Garissa the
  U-Net scores 0.215 with a pass three days before the 29 April reference and
  0.581 with one nine days after it, while the Tana was still rising.
- The best extent model puts only 3 to 14% of UNOSAT's affected buildings in
  its masks; a fixed change-detection rule comes within the spread expected
  from the building data alone.
- Along the lower Tana, flood water peaks at 535 km² on 20 May on the middle
  river and about four weeks later near the delta.

The tables behind these numbers are in `outputs/` and `models/`; the
reasoning behind every analytical choice is in `DECISIONS.md`.

## What is in the repository

```
data/
  reference/      UNOSAT product catalogue and rasterised flood water per AOI
  aoi/            AOI polygons, Tana corridor tiles, Sentinel-1 passes and day gaps
  cache/          (gitignored) raw imagery, Sen1Floods11, stacks, ancillary layers
pipeline/         every stage, runnable as scripts or from the notebook
kaggle/           GPU kernels that trained the U-Nets
models/           trained U-Net weights and Sen1Floods11 test results
outputs/          metrics, flood masks per AOI and method, Tana time series
notebook/         kenya_floods_sar.ipynb, regenerates every number and figure
paper/            LaTeX source, figures and PDF
index.html        landing page
```

## Pipeline

| Stage | Script | Output |
|---|---|---|
| UNOSAT catalogue and archive | `fetch_reference.py` | `data/reference/catalogue.csv` |
| Tana and Galana corridor tiles | `tana_corridor.py` | `data/aoi/tana_tiles.gpkg` |
| Rasterised reference | `reference.py` | `data/reference/*_flood.tif` |
| Sentinel-1 passes | `pick_scenes.py` | `data/aoi/scenes.csv` |
| GRD to sigma0 | `s1_grd.py` | used by every later stage |
| Check against Earth Engine | `validate_grd.py` | `outputs/grd_vs_gee_sen1floods11.csv` |
| Sen1Floods11 and pre-event chips | `fetch_sen1floods11.py`, `pre_event_chips.py` | `data/sen1floods11_*.csv` |
| U-Net training | `train_unet.py` (Kaggle) | `models/` |
| Input stacks | `prepare.py`, `ancillary.py` | `data/cache/stacks/` |
| Flood masks | `methods.py`, `classical.py` | `outputs/masks/` |
| Agreement with UNOSAT | `evaluate.py`, `metrics.py` | `outputs/metrics_*.csv` |
| Day-gap sensitivity | `sensitivity.py` | `outputs/day_gap_sensitivity.csv` |
| Building exposure | `exposure.py` | `outputs/exposure.csv` |
| Lower Tana time series | `tana_timeseries.py` | `outputs/tana_timeseries.csv`, `outputs/tana/` |
| Figures | `figures.py` | `paper/fig_*.png` |

## Reproducing

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook/kenya_floods_sar.ipynb
```

Every input is public and needs no account. UNOSAT products come from the
Humanitarian Data Exchange; Sentinel-1, Sentinel-2, the Copernicus DEM, JRC
Global Surface Water and ESA WorldCover come from Microsoft Planetary
Computer; Sen1Floods11 comes from its public Google Cloud bucket; HAND comes
from ASF's GLO-30 HAND on AWS; and Google Open Buildings v3 comes from its
public bucket.

A first run downloads about 10 GB (mostly Sen1Floods11 and the Open Buildings
tile) and fills about 17 GB of `data/cache/`, most of it per-AOI input stacks. The
U-Nets were trained on a Kaggle T4 GPU with the kernels in `kaggle/`, and the
notebook uses the weights committed in `models/`. Set `TRAIN_LOCALLY = True`
to retrain them on your own hardware.

## Licences

Code is MIT ([`LICENSE`](LICENSE)). Flood rasters, tables and figures
produced here are CC BY 4.0, and the UNOSAT reference data and their rasterised
derivatives remain CC BY-SA ([`LICENSE-DATA`](LICENSE-DATA)).

This is independent research. It is not produced, reviewed, commissioned or
endorsed by UNOSAT, UNITAR, ESA, the Copernicus programme, the National
Drought Management Authority, the Kenya Space Agency or the Kenya Red Cross.
