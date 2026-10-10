"""Figures for the paper, in the portfolio palette.

Flood water is blue, permanent water navy, omission amber and commission grey;
backgrounds are greyscale SAR or plain light grey. Methods sit on an axis
rather than in colour, and magnitude uses a single-hue navy ramp.
"""

import geopandas as gpd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

from boundaries import counties
from config import (BG_ALT, BLUE, BLUE_LT, BORDER, COMMISSION, MUTED, NAVY, OMISSION, OUTPUTS, PAPER,
                    PERMANENT_WATER_OCCURRENCE, TEXT)
from evaluate import load_mask
from prepare import stack
from reference import SPEC, UTM, extent

NAVY_RAMP = LinearSegmentedColormap.from_list("navy_ramp", [BG_ALT, BLUE_LT, "#9EC1E8", "#3C6FA5", NAVY])

LABELS = {
    "otsu_vv": "Otsu, VV",
    "otsu_vh": "Otsu, VH",
    "change_otsu_vv": "Change, Otsu, VV",
    "change_otsu_vh": "Change, Otsu, VH",
    "change_fixed_vv": "Change, -3 dB, VV",
    "change_fixed_vh": "Change, -3 dB, VH",
    "unet_post": "U-Net, post",
    "unet_post_hand": "U-Net, post, hand only",
    "unet_pre": "U-Net, pre + post",
}
GROUPS = {"riverine": "Riverine and rural", "urban": "Urban and peri-urban"}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"],
        "font.size": 9,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT,
        "axes.titlesize": 10,
        "axes.grid": False,
        "grid.color": "#E6EDF4",
        "grid.linewidth": 0.6,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def ordered_methods(df: pd.DataFrame) -> list[str]:
    return [m for m in LABELS if m in set(df["method"])]


def kenya_map() -> None:
    c = counties()
    aois = gpd.GeoDataFrame({"aoi": list(SPEC)}, geometry=[extent(s) for s in SPEC.values()], crs=UTM).to_crs(4326)
    fig, ax = plt.subplots(figsize=(6.2, 6.6))
    c.plot(ax=ax, color="#FFFFFF", edgecolor=BORDER, linewidth=0.5)
    for name in ("Garissa", "Tana River", "Nairobi", "Kiambu", "Machakos"):
        c[c["county"] == name].plot(ax=ax, color=BG_ALT, edgecolor=MUTED, linewidth=0.6)
    tiles = aois[aois["aoi"].str.startswith("tana_")]
    tiles.plot(ax=ax, color="#9EC1E8", edgecolor="none")
    points = aois[~aois["aoi"].str.startswith("tana_")]
    points.plot(ax=ax, color=NAVY, edgecolor=NAVY, linewidth=0.8)
    for name, (xy, text_xy, ha) in {"Nairobi, Kiambu, Machakos": ((36.95, -1.2), (35.6, -2.3), "center"),
                                    "Garissa": ((39.65, -0.45), (40.2, -0.1), "left"),
                                    "Tana and Galana corridor": ((40.2, -2.3), (40.9, -1.5), "left")}.items():
        ax.annotate(name, xy, xytext=text_xy, fontsize=8, color=TEXT, ha=ha,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
    ax.set_xlim(33.8, 42.2)
    ax.set_ylim(-4.9, 5.2)
    ax.set_axis_off()
    ax.legend(handles=[Patch(color=NAVY, label="Garissa and urban AOIs (UNOSAT extent)"),
                       Patch(color="#9EC1E8", label="Tana and Galana corridor tiles"),
                       Patch(facecolor=BG_ALT, edgecolor=MUTED, label="Counties named in the text")],
              loc="lower left", fontsize=7.5)
    fig.savefig(PAPER / "fig_aoi_map.png")
    plt.close(fig)


def error_panels(aoi: str, method: str, out: str) -> None:
    s = stack(aoi)
    pred = load_mask(aoi, method) == 1
    ref = s["reference"] == 1
    valid = (s["reference"] != 255) & (load_mask(aoi, method) != 255)
    permanent = s["jrc_occurrence"] > PERMANENT_WATER_OCCURRENCE
    sar = dict(cmap="gray", vmin=-27, vmax=-8)

    def overlay(ax, mask, color):
        ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([color]), interpolation="nearest")

    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
    axes[0].imshow(s["pre_vh"], **sar)
    axes[0].set_title("Pre-event VH")
    axes[1].imshow(s["post_vh"], **sar)
    axes[1].set_title("Post-event VH")
    for ax, mask, title in ((axes[2], ref, "UNOSAT flood water"), (axes[3], pred, LABELS[method])):
        ax.imshow(np.where(valid, 1.0, 0.0), cmap=ListedColormap(["#E5E7EB", "#FFFFFF"]), interpolation="nearest")
        overlay(ax, mask & valid, BLUE)
        overlay(ax, permanent, NAVY)
        ax.set_title(title)
    axes[4].imshow(np.where(valid, 1.0, 0.0), cmap=ListedColormap(["#E5E7EB", "#FFFFFF"]), interpolation="nearest")
    overlay(axes[4], pred & ref & valid, BLUE)
    overlay(axes[4], ~pred & ref & valid, OMISSION)
    overlay(axes[4], pred & ~ref & valid, COMMISSION)
    axes[4].set_title("Agreement")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(handles=[Patch(color=BLUE, label="Flood water, agreed"), Patch(color=OMISSION, label="Omission"),
                        Patch(color=COMMISSION, label="Commission"), Patch(color=NAVY, label="Permanent water (JRC)"),
                        Patch(color="#E5E7EB", label="Outside evaluation domain")],
               loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(PAPER / out)
    plt.close(fig)


def iou_by_method() -> None:
    df = pd.read_csv(OUTPUTS / "metrics_by_group.csv")
    df = df[~df["hand_mask"]]
    methods = ordered_methods(df)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, (group, title) in zip(axes, GROUPS.items()):
        g = df[df["group"] == group].set_index("method").loc[methods]
        y = np.arange(len(methods))[::-1]
        ax.hlines(y, g["iou_lo"], g["iou_hi"], color=MUTED, linewidth=1.6)
        ax.plot(g["iou"], y, "o", color=NAVY, markersize=6)
        for yi, v in zip(y, g["iou"]):
            ax.annotate(f"{v:.2f}", (v, yi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=TEXT)
        ax.set_title(title)
        ax.set_xlim(0, max(0.6, df["iou_hi"].max() + 0.05))
        ax.set_xlabel("IoU, flood class (95% block bootstrap interval)")
        ax.grid(axis="x")
        ax.set_yticks(y, [LABELS[m] for m in methods])
    fig.tight_layout()
    fig.savefig(PAPER / "fig_iou_by_method.png")
    plt.close(fig)


def landcover_heatmap() -> None:
    df = pd.read_csv(OUTPUTS / "metrics_by_landcover.csv")
    df = df[~df["hand_mask"] & (df["worldcover"] != "Permanent water")]
    methods = ordered_methods(df)
    classes = ["Tree cover", "Shrubland", "Grassland", "Cropland", "Built-up", "Bare", "Herbaceous wetland"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, (group, title) in zip(axes, GROUPS.items()):
        g = df[df["group"] == group].pivot_table(index="method", columns="worldcover", values="f1")
        cols = [c for c in classes if c in g.columns]
        g = g.reindex(index=methods, columns=cols)
        im = ax.imshow(g.to_numpy(), cmap=NAVY_RAMP, vmin=0, vmax=0.8, aspect="auto")
        for (i, j), v in np.ndenumerate(g.to_numpy()):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#FFFFFF" if v > 0.45 else TEXT)
        ref_km2 = df[(df["group"] == group) & (df["method"] == methods[0])].set_index("worldcover")["ref_km2"]
        ax.set_xticks(range(len(cols)), [f"{c}\n{ref_km2.get(c, 0):.1f} km² flooded" for c in cols],
                      rotation=35, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods])
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8, label="F1, flood class")
    fig.savefig(PAPER / "fig_f1_landcover.png")
    plt.close(fig)


def exposure() -> None:
    """Ratio of buildings in each mask to UNOSAT's count, one shared log axis.

    The band spans the ratios obtained with UNOSAT's own flood extent: the
    disagreement that comes from the building data alone."""
    df = pd.read_csv(OUTPUTS / "exposure.csv")
    names = {"garissa_s2": "Garissa, 29 Apr", "garissa_pl": "Garissa, 5 May", "nairobi_pl": "Nairobi area, 1 May"}
    methods = ordered_methods(df)
    band = df[df["method"] == "unosat_extent"]["ratio_to_unosat"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.axvspan(band.min(), band.max(), color=BG_ALT, zorder=0)
    ax.axvline(1, color=OMISSION, linewidth=1.2, linestyle="--", zorder=1)
    y = np.arange(len(methods))[::-1]
    markers = {"garissa_s2": ("o", NAVY), "garissa_pl": ("s", BLUE), "nairobi_pl": ("D", MUTED)}
    for aoi, label in names.items():
        g = df[df["aoi"] == aoi].set_index("method").reindex(methods)
        m, c = markers[aoi]
        ax.plot(g["ratio_to_unosat"], y, m, color=c, markersize=7, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 20)
    ax.set_xticks([0.03, 0.1, 0.3, 1, 3, 10], ["0.03", "0.1", "0.3", "1", "3", "10"])
    ax.set_yticks(y, [LABELS[m] for m in methods])
    ax.set_xlabel("Open Buildings in flood mask / UNOSAT structure count")
    ax.grid(axis="x")
    ax.set_ylim(-0.7, len(methods) - 0.1)
    ax.annotate("spread from building data alone", (band.max(), len(methods) - 0.45), xytext=(4, 0),
                textcoords="offset points", fontsize=7.5, color=MUTED, va="center")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(PAPER / "fig_exposure.png")
    plt.close(fig)


def day_gap() -> None:
    df = pd.read_csv(OUTPUTS / "day_gap_sensitivity.csv")
    names = {"garissa_s2": "Garissa, 29 Apr reference", "garissa_pl": "Garissa, 5 May reference",
             "nairobi_pl": "Nairobi area, 1 May reference"}
    methods = [m for m in ("change_fixed_vv", "otsu_vh", "unet_post") if m in set(df["method"])]
    styles = {"change_fixed_vv": (NAVY, "-", "o"), "otsu_vh": (MUTED, "--", "s"), "unet_post": (BLUE, "-", "D")}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (aoi, title) in zip(axes, names.items()):
        for m in methods:
            g = df[(df["aoi"] == aoi) & (df["method"] == m)].sort_values("day_gap")
            color, ls, marker = styles[m]
            ax.plot(g["day_gap"], g["iou"], ls, marker=marker, color=color, linewidth=2, markersize=6, label=LABELS[m])
        ax.axvline(0, color=BORDER, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Sentinel-1 date minus reference date (days)")
        ax.grid(axis="y")
    axes[0].set_ylabel("IoU, flood class")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(PAPER / "fig_day_gap.png")
    plt.close(fig)


def tana_series() -> None:
    df = pd.read_csv(OUTPUTS / "tana_timeseries.csv", parse_dates=["date"])
    orbits = sorted(df["orbit"].unique())
    fig, axes = plt.subplots(1, len(orbits), figsize=(5 * len(orbits), 3.4), squeeze=False)
    for ax, orbit in zip(axes[0], orbits):
        for county, color in (("Garissa", NAVY), ("Tana River", BLUE)):
            g = df[(df["orbit"] == orbit) & (df["county"] == county)].sort_values("date")
            if g["observed_km2"].sum() == 0:
                continue
            ax.plot(g["date"], g["flooded_km2"], "-o", color=color, linewidth=2, markersize=6)
            ax.annotate(f"{county} ({g['observed_km2'].median():,.0f} km² observed)", (g["date"].iloc[-1], g["flooded_km2"].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=TEXT, va="center")
        ax.set_title(f"Relative orbit {orbit}")
        ax.set_ylabel("Flood water (km²)")
        ax.grid(axis="y")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("1 %b"))
    fig.tight_layout()
    fig.savefig(PAPER / "fig_tana_series.png")
    plt.close(fig)


def main() -> None:
    apply_style()
    kenya_map()
    iou_by_method()
    landcover_heatmap()
    exposure()
    day_gap()
    if (OUTPUTS / "tana_timeseries.csv").exists():
        tana_series()


if __name__ == "__main__":
    main()
