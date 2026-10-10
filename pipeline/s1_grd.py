"""Sentinel-1 GRD to terrain-corrected sigma0 on a UTM grid.

Planetary Computer's RTC collection needs an account, so scenes are processed
from its GRD collection instead, following the steps Earth Engine applies to
build the Sen1Floods11 chips: thermal noise removal, calibration to sigma0, and
orthorectification against a DEM, with no radiometric terrain flattening. Doing
the same here keeps Kenya scenes in the radiometric domain the U-Net was
trained on.
"""

import time
from dataclasses import dataclass
from functools import lru_cache
import xml.etree.ElementTree as ET

import numpy as np
import planetary_computer
import pystac
import rasterio
import requests
from pyproj import Transformer
from rasterio.windows import Window
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

POLS = ("vv", "vh")


@dataclass
class Grid:
    crs: str
    transform: rasterio.Affine
    width: int
    height: int

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        cols, rows = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        return self.transform * (cols, rows)


@lru_cache(maxsize=64)
def _xml(href: str) -> ET.Element:
    r = requests.get(_fresh(href), timeout=120)
    r.raise_for_status()
    return ET.fromstring(r.content)


def _floats(node: ET.Element, tag: str) -> np.ndarray:
    return np.array(node.find(tag).text.split(), dtype=float)


def _product_annotation(item: pystac.Item, pol: str) -> str:
    # On recent IPF versions the catalogue's schema-product asset points at the
    # RFI annotation; the product annotation sits beside the measurement.
    href = item.assets[pol].href.split("?")[0]
    return href.replace("/measurement/", "/annotation/").replace(".tiff", ".xml")


def geolocation_grid(item: pystac.Item, pol: str) -> dict[str, np.ndarray]:
    root = _xml(_product_annotation(item, pol))
    pts = root.findall(".//geolocationGridPoint")
    keys = ("line", "pixel", "latitude", "longitude", "height", "incidenceAngle")
    grid = {k: np.array([float(p.find(k).text) for p in pts]) for k in keys}
    grid["range_spacing"] = float(root.find(".//rangePixelSpacing").text)
    return grid


def _poly_terms(x: np.ndarray, y: np.ndarray, degree: int = 3) -> np.ndarray:
    return np.stack([x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)], axis=-1)


def _fit(src_x, src_y, values, x, y):
    cx, cy, s = src_x.mean(), src_y.mean(), max(np.ptp(src_x), np.ptp(src_y))
    coef, *_ = np.linalg.lstsq(_poly_terms((src_x - cx) / s, (src_y - cy) / s), values, rcond=None)
    return _poly_terms((x - cx) / s, (y - cy) / s) @ coef


def radar_coordinates(item: pystac.Item, grid: Grid, ellipsoid_height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each target pixel to fractional (line, pixel) in the GRD image.

    The GRD is projected to the ellipsoid raised by the annotation's terrain
    height, which is constant in range. A target point whose true height
    differs from that annotated height is imaged displaced in ground range by
    dh / tan(incidence); the pixel coordinate is shifted by that amount.
    Getting this sign wrong moves Nairobi pixels by more than 100 m.
    """
    g = geolocation_grid(item, "vv")
    lon, lat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(*grid.centres())

    near = (
        (g["longitude"] > lon.min() - 0.6) & (g["longitude"] < lon.max() + 0.6)
        & (g["latitude"] > lat.min() - 0.6) & (g["latitude"] < lat.max() + 0.6)
    )
    gx, gy = g["longitude"][near], g["latitude"][near]
    line = _fit(gx, gy, g["line"][near], lon, lat)
    pixel = _fit(gx, gy, g["pixel"][near], lon, lat)
    h_ref = _fit(gx, gy, g["height"][near], lon, lat)
    inc = np.deg2rad(_fit(gx, gy, g["incidenceAngle"][near], lon, lat))

    if np.corrcoef(g["pixel"], g["incidenceAngle"])[0, 1] <= 0:
        raise ValueError("expected pixel index to increase with incidence angle (near to far range)")
    pixel -= (ellipsoid_height - h_ref) / np.tan(inc) / g["range_spacing"]
    return line, pixel


def _calibration(item: pystac.Item, pol: str, lines: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    root = _xml(item.assets[f"schema-calibration-{pol}"].href)
    vecs = root.findall(".//calibrationVector")
    vec_lines = np.array([int(v.find("line").text) for v in vecs])
    vec_pixels = _floats(vecs[0], "pixel")
    table = np.stack([np.interp(vec_pixels, _floats(v, "pixel"), _floats(v, "sigmaNought")) for v in vecs])
    interp = RegularGridInterpolator((vec_lines, vec_pixels), table, bounds_error=False, fill_value=None)
    ll, pp = np.meshgrid(lines, pixels, indexing="ij")
    return interp((ll, pp))


def _noise(item: pystac.Item, pol: str, lines: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    root = _xml(item.assets[f"schema-noise-{pol}"].href)
    # Products before IPF 2.9 (most of Sen1Floods11) carry a range-only noise LUT.
    legacy = not root.findall(".//noiseRangeVector")
    rvecs = root.findall(".//noiseVector" if legacy else ".//noiseRangeVector")
    lut = "noiseLut" if legacy else "noiseRangeLut"
    rv_lines = np.array([int(v.find("line").text) for v in rvecs])
    rng = np.stack([np.interp(pixels, _floats(v, "pixel"), _floats(v, lut)) for v in rvecs])
    rng = np.stack([np.interp(lines, rv_lines, rng[:, j]) for j in range(len(pixels))], axis=1)

    az = np.ones_like(rng)
    for block in root.findall(".//noiseAzimuthVector"):
        l0, l1 = int(block.find("firstAzimuthLine").text), int(block.find("lastAzimuthLine").text)
        p0, p1 = int(block.find("firstRangeSample").text), int(block.find("lastRangeSample").text)
        rows = (lines >= l0) & (lines <= l1)
        cols = (pixels >= p0) & (pixels <= p1)
        if rows.any() and cols.any():
            lut = np.interp(lines[rows], _floats(block, "line"), _floats(block, "noiseAzimuthLut"))
            az[np.ix_(rows, cols)] = lut[:, None]
    return rng * az


def _fresh(href: str) -> str:
    """Re-sign with a current SAS token: items signed at search time carry a
    token that expires during long runs, after which reads fail with 403."""
    return planetary_computer.sign(href.split("?")[0])


def _read_window(href: str, window: Window, attempts: int = 4) -> np.ndarray:
    """Windowed read of a remote COG, retried because long runs meet transient
    network failures."""
    for attempt in range(attempts):
        try:
            with rasterio.open(_fresh(href)) as src:
                return src.read(1, window=window).astype("float64")
        except rasterio.errors.RasterioIOError:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))


def sigma0_pass(items: list[pystac.Item], grid: Grid, ellipsoid_height: np.ndarray) -> dict[str, np.ndarray]:
    """sigma0 from consecutive slices of one pass, each filling the other's gaps."""
    out = {pol: np.full((grid.height, grid.width), np.nan, dtype="float32") for pol in POLS}
    for item in items:
        try:
            s0 = sigma0(item, grid, ellipsoid_height)
        except ValueError:
            continue
        for pol in POLS:
            out[pol] = np.where(np.isnan(out[pol]), s0[pol], out[pol])
    return out


def sigma0(item: pystac.Item, grid: Grid, ellipsoid_height: np.ndarray) -> dict[str, np.ndarray]:
    """Return sigma0 in dB for VV and VH on `grid`; NaN outside the swath."""
    line, pixel = radar_coordinates(item, grid, ellipsoid_height)
    out = {}
    with rasterio.open(_fresh(item.assets["vv"].href)) as src:
        n_lines, n_pixels = src.height, src.width
    r0 = int(max(np.floor(np.nanmin(line)) - 2, 0))
    r1 = int(min(np.ceil(np.nanmax(line)) + 3, n_lines))
    c0 = int(max(np.floor(np.nanmin(pixel)) - 2, 0))
    c1 = int(min(np.ceil(np.nanmax(pixel)) + 3, n_pixels))
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"{item.id} does not cover the target grid")
    lines, pixels = np.arange(r0, r1), np.arange(c0, c1)
    for pol in POLS:
        dn = _read_window(item.assets[pol].href, Window(c0, r0, c1 - c0, r1 - r0))
        cal = _calibration(item, pol, lines, pixels)
        power = (dn**2 - _noise(item, pol, lines, pixels)) / cal**2
        power[dn == 0] = np.nan
        power = np.clip(power, 1e-5, None)
        sampled = map_coordinates(power, [line - r0, pixel - c0], order=1, mode="constant", cval=np.nan)
        outside = (line < 0) | (line > n_lines - 1) | (pixel < 0) | (pixel > n_pixels - 1)
        sampled[outside] = np.nan
        out[pol] = (10 * np.log10(sampled)).astype("float32")
    return out
