# Decisions log

Every choice the operating brief left open, and every departure from it, with
the reason and where it is implemented. Numbers quoted here are read from files
in this repository.

## Reference data

**One cumulative UNOSAT archive.** Every FL20240426KEN product page on HDX
links a shapefile package of the same name. The packages are cumulative: the
nine dated 9 May are byte-identical in size (59,835,613 bytes) and the two dated
10 May are a superset of them and of the 1 May national product. Only the most
recent package is downloaded (`pipeline/fetch_reference.py`), and each product's
reported figures are parsed from its HDX description into
`data/reference/catalogue.csv`.

**Raw shapefiles are not committed.** The package is 60 MB and contains
single-feature multipolygons of up to 27 MB. The fetcher re-downloads it
unmodified; what is committed is the rasterised reference per AOI
(`data/reference/*_flood.tif`), which remains a CC BY-SA derivative.

**AOIs.** The archive defines fewer analyses than HDX lists products:

| AOI | Reference | Notes |
|---|---|---|
| `garissa_s2` | Sentinel-2, 29 April 2024 | No Garissa-specific flood layer exists in the archive. The flood layer of the Tana and Galana product from the same image, clipped to the Garissa analysis extent, gives 27.9 km² against the 26 km² UNOSAT reports for Garissa. |
| `garissa_pl` | Pléiades, 5 May 2024 | 21.7 km² against 22 km² reported. The structure layer's sensor field says WorldView-2 while the extent and flood layers say Pléiades; the HDX description says Pléiades. |
| `nairobi_pl` | Pléiades, 1 May 2024 | The four urban products (Nairobi; Nairobi and Kiambu; Nairobi and Machakos; Kiambu) share one analysis extent, one flood layer and one structure layer. Their structure counts (3,387, 3,663, 4,246, 1,815) sum to exactly the 13,111 points in that layer, so they are one AOI here. |
| `tana_*` (70 tiles) | Sentinel-2, 29 April 2024 | The Tana and Galana product (70,874 km² analysed) has no HDX page of its own. It is evaluated along a river corridor, see below. |

Excluded: the national NOAA-VIIRS product of 24 to 28 April (375 m pixels,
not comparable at 10 m), the Mai Mahiu and Kijabe landslide product (not flood
water), and the three cropland products (subsets of the urban analysis).

**Cloud.** UNOSAT's cloud-obstruction polygons are removed from the
evaluation domain: 73 km² of the 580 km² urban extent, 0.5 km² at Garissa on
29 April, and 8,690 km² of the Tana and Galana extent. UNOSAT could not see the
ground there, so neither agreement nor disagreement can be scored.

**Flood water only.** Every UNOSAT flood layer used is classed "Flood Water,
New Water / Water Increase", separate from the Garissa and Tana "Water Extent"
layers that include the river. JRC permanent water (occurrence above 80%) is
also removed from both prediction and evaluation domain, as the brief
specifies. At Garissa this removes nothing: JRC occurrence peaks at 72% on the
Tana there, since the channel migrates and carries sediment.

**Tana and Galana corridor.** Processing 70,874 km² at 10 m was not practical,
and evaluating on the whole extent would score mostly dry land. The corridor
is defined from JRC water, not from the reference: land within 5 km of any
300 m cell where JRC occurrence reaches 50%, cut into 20 km UTM tiles, minus the
Garissa extent (scored separately) and tiles under 20 km²
(`pipeline/tana_corridor.py`). It holds 8,785 km² and 361 of the 469 km² of
flood water UNOSAT mapped in that product. The rule also picks up small
reservoirs and pans away from the rivers; they are kept rather than edited by
hand.

## Sentinel-1 processing

**GRD, not RTC.** Planetary Computer's `sentinel-1-rtc` collection now states
that an account is required for SAS tokens. Following the brief's fallback,
scenes come from `sentinel-1-grd` and are processed here
(`pipeline/s1_grd.py`): thermal noise removal, calibration to sigma0 from the
annotation LUTs, and geocoding from the annotation geolocation grid with a
per-pixel range shift of (h - h_annotation) / tan(incidence) for terrain
height. There is no radiometric terrain flattening. This is the same product
Earth Engine produces and from which Sen1Floods11 was built, so Kenya inputs
sit in the U-Net's training domain.

**Validated against Earth Engine.** On 18 Sen1Floods11 hand-labelled chips
from 9 events, this processing reproduces the Earth Engine chips with a median
bias between -0.02 and +0.04 dB and a median correlation of 0.97 at zero shift
(`outputs/grd_vs_gee_sen1floods11.csv`). Best-fit offsets are zero for most
chips and one pixel for a few; the Mekong chips are offset two to three pixels,
which is attributed to Earth Engine's use of SRTM against the Copernicus DEM
here. Pakistan and Paraguay chips could not be tested because no single scene
from the event date covered them.

**Ellipsoidal heights.** Copernicus DEM heights are EGM2008 orthometric while
the GRD annotation is ellipsoidal. The geoid undulation is -27 m at Garissa and
-16 m at Nairobi, two to four range pixels if ignored, and is applied with
PROJ's EGM2008 grid.

**Catalogue quirk.** On recent products the catalogue's `schema-product-*`
asset points at the RFI annotation; the product annotation is read from beside
the measurement file instead.

**Passes, not slices.** An AOI can straddle two consecutive slices of one
datatake, so slices are grouped into passes and mosaicked. A pass must image at
least half of an AOI; corridor tiles that straddle two tracks keep the larger
share and the unimaged part drops out as no-data.

**Scene selection** (`data/aoi/scenes.csv`). Post-event: the pass closest to
the reference date. Every point AOI has one within three days: Garissa 29 April
is matched by 26 April (-3), Garissa 5 May by 8 May (+3), Nairobi 1 May by
1 May (0). Garissa is covered by one track only (relative orbit 159, every 12
days since Sentinel-1B failed). Pre-event: the latest January to February 2024
pass on the same relative orbit, 26 February for Garissa and 19 February for
Nairobi, to give the late-2023 short-rains flooding the longest time to
recede.

## Classical methods

`pipeline/classical.py`. Thresholds are fitted as in the Sen1Floods11
reference classification: Otsu on the pooled histogram of the 10% most
heterogeneous 1 km tiles (at least five), applied to a 3 x 3 focal mean. The
smoothing kernel size is not stated in Bonafilia et al. (2020); 3 x 3 is the
smallest that suppresses single-pixel speckle. The change-detection Otsu
threshold is capped at 0 dB, so a unimodal ratio histogram cannot label
brightening as flooding; in Nairobi the fitted value was +0.2 dB for VH and
the cap applied.

## Training data and the U-Net

**Sen1Floods11 events** (`data/sen1floods11_events.csv`). Three of the eleven
flood events are African: Ghana (53 hand-labelled chips), Nigeria (18) and
Somalia (26). Somalia (May 2018, Juba and Shabelle) is the only East African
event, so the transfer to Kenya is out of region for most but not all of the
training data. Hand-labelled chips of the Cambodia event are prefixed
"Mekong"; Colombia has weakly labelled chips only.

**Bolivia held out completely.** Bolivia is the paper's out-of-distribution
test event, yet the bucket includes Bolivia chips in the weakly labelled set.
They are excluded from training here.

**Weak labels.** The paper's best setting for flood water is weak labels
derived from Sentinel-2 (`S2IndexLabelWeak`), plus the hand-labelled training
split.

**Architecture and schedule.** The brief specifies a U-Net; the paper used an
FCN with a ResNet-50 backbone. The U-Net here has four levels, 32 base
channels, and GroupNorm in place of BatchNorm, mirroring the paper's
substitution. Inputs are scaled exactly as in the reference code (clip to
-50..1 dB, rescale, standardise with the published means and deviations),
except that missing pixels become -50 dB rather than 0 dB, since 0 dB is a
plausible bright-target value. Loss, optimiser, crop and schedule follow the
paper. Training runs on a Kaggle GPU (`kaggle/`), pulling Sen1Floods11 from its
public bucket inside the kernel, after a local timing showed about 99 minutes
per epoch on this machine's CPU.

**Pre-event U-Net.** Sen1Floods11 ships only the flood-date image. Pre-event
chips are built here for the hand-labelled set with the validated GRD pipeline
(`pipeline/pre_event_chips.py`): the latest pass on the event's relative orbit
at least 30 days before the event, searching 120 days back. Weakly labelled
chips are not paired, which would mean processing 4,385 more chips, so this
model and its post-only control are both trained on hand-labelled chips that
have a pre-event pair, and differ only in input channels.

## Optical comparator

**Omitted.** The Prithvi-EO-1.0-100M Sen1Floods11 checkpoint exists on Hugging
Face under Apache-2.0, but it takes six Sentinel-2 bands. Within six days of
each reference date: Nairobi's three Sentinel-2 acquisitions are 52%, 89% and
92% cloud; Garissa's only usable acquisition for 29 April is the image UNOSAT
built that reference from, which would make the comparison circular; and for
5 May the choice is 4 May at 100% cloud or 9 May at 36%. This is the reason the
brief gives for using SAR at all.

## Ancillary data

**HAND** is taken from ASF's global GLO-30 HAND, computed from the same
Copernicus DEM, rather than recomputed. **Ancillary stacks are not committed**:
they are tens of megabytes per AOI and regenerate from public sources.

## Exposure

Google Open Buildings v3 is used as points (centroid, footprint area,
confidence) from S2 cell 183, which covers all AOIs. A building counts as
exposed when its centroid falls in a flood pixel inside the evaluation domain.
Buildings with confidence below 0.70 are dropped. UNOSAT's structure layers are
also points, digitised on very high resolution imagery, so a third count, Open
Buildings inside UNOSAT's own flood extent, separates disagreement in the
building data from disagreement in the flood extent.
