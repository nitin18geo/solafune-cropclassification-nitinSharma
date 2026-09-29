"""Exploratory data analysis: statistics, plotting helpers and figures for the report.

Contains the PASTIS colour map, RGB composites and all EDA plots. Every plotting
function returns the matplotlib Figure and saves it as a PNG if `path` is given.
Streams through the patches one at a time (about 15 MB each) so memory stays low.

Run all EDA and save tables to outputs/metrics and figures to outputs/figures:
    python -m src.eda --config configs/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import src
import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")  # no display needed when run as a script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from tqdm import tqdm

from src.data_loading import list_patch_ids, load_config, load_metadata, load_patch

REFL_SCALE = 10000.0
RGB_BANDS = (2, 1, 0)  # B4 (red), B3 (green), B2 (blue)
BRIGHT_BLUE_THRESHOLD = 0.2  # B2 reflectance above this is treated as likely cloud/haze



# --------------------------------------------------------------------------
# Colours and basic rendering
# --------------------------------------------------------------------------
def pastis_cmap(n_classes: int = 20) -> tuple[ListedColormap, BoundaryNorm]:
    """Colour map matching the PASTIS legend: black background, tab20 crops, white void."""
    tab20 = plt.get_cmap("tab20").colors
    colors = ["black"] + [tab20[i] for i in range(1, n_classes - 1)] + ["white"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n_classes + 0.5), cmap.N)
    return cmap, norm


def class_legend(classes: dict, present: list[int] | None = None) -> list[Patch]:
    """Legend handles for the given class IDs (all classes if `present` is None)."""
    cmap, _ = pastis_cmap(len(classes))
    ids = sorted(present) if present is not None else sorted(classes)
    return [
        Patch(facecolor=cmap(int(c)), edgecolor="grey", label=f"{c}: {classes[int(c)]}")
        for c in ids
    ]


def rgb_composite(s2_date: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    """True-colour image (H, W, 3) from one date of shape (10, H, W).

    Uses B4/B3/B2, converts to reflectance, and applies a percentile stretch
    so that dull agricultural scenes are visible.
    """
    rgb = s2_date[list(RGB_BANDS)].astype(np.float32) / REFL_SCALE
    rgb = np.clip(rgb, 0, 1)
    lo, hi = np.percentile(rgb, (pmin, pmax))
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.transpose(rgb, (1, 2, 0))


def save_fig(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    """Save a figure to disk if a path is given, creating folders as needed."""
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# EDA figures
# --------------------------------------------------------------------------
def plot_class_distribution(dist: pd.DataFrame, classes: dict, path=None) -> plt.Figure:
    """Bar chart of pixel share per class (log scale) and patches containing each class."""
    cmap, _ = pastis_cmap(len(classes))
    colors = [cmap(int(c)) for c in dist.index]
    labels = [f"{c}: {classes[int(c)]}" for c in dist.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes[0].barh(labels, dist["pixel_share_pct"], color=colors, edgecolor="grey")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Share of all pixels (%, log scale)")
    axes[0].set_title("Pixel distribution by class")
    axes[0].invert_yaxis()

    axes[1].barh(labels, dist["n_patches"], color=colors, edgecolor="grey")
    axes[1].set_xlabel("Number of patches containing the class")
    axes[1].set_title("Class presence across patches")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_class_by_fold(share: pd.DataFrame, classes: dict, path=None) -> plt.Figure:
    """Heatmap of each class's pixel share within each fold (columns sum to 100%)."""
    fig, ax = plt.subplots(figsize=(8, 9))
    data = share.values
    im = ax.imshow(np.log10(data + 0.01), cmap="viridis", aspect="auto")
    ax.set_xticks(range(share.shape[1]), [f"Fold {f}" for f in share.columns])
    ax.set_yticks(range(share.shape[0]), [f"{c}: {classes[int(c)]}" for c in share.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = "0" if val == 0 else (f"{val:.1f}" if val >= 0.1 else "<0.1")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if val < 1 else "black")
    ax.set_title("Pixel share (%) of each class within each fold")
    fig.colorbar(im, ax=ax, label="log10(share %)", shrink=0.6)
    fig.tight_layout()
    return save_fig(fig, path)


def plot_label_examples(labels: dict, classes: dict, ncols: int = 3, path=None) -> plt.Figure:
    """Grid of label maps for selected patches, with a shared legend."""
    cmap, norm = pastis_cmap(len(classes))
    n = len(labels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    present = set()
    for ax, (pid, y) in zip(axes, labels.items()):
        ax.imshow(y, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"Patch {pid}")
        present |= set(np.unique(y).tolist())
    for ax in axes:
        ax.axis("off")
    fig.legend(handles=class_legend(classes, sorted(present)),
               loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout()
    return save_fig(fig, path)


def plot_rgb_label_pairs(items: list, classes: dict, date_label: str, path=None) -> plt.Figure:
    """Rows of [RGB composite | label map]. `items` = [(patch_id, s2_date, label), ...]."""
    cmap, norm = pastis_cmap(len(classes))
    fig, axes = plt.subplots(len(items), 2, figsize=(8, 4 * len(items)))
    axes = np.atleast_2d(axes)
    present = set()
    for row, (pid, s2_date, y) in zip(axes, items):
        row[0].imshow(rgb_composite(s2_date))
        row[0].set_title(f"Patch {pid}: RGB ({date_label})")
        row[1].imshow(y, cmap=cmap, norm=norm, interpolation="nearest")
        row[1].set_title(f"Patch {pid}: labels")
        present |= set(np.unique(y).tolist())
        for ax in row:
            ax.axis("off")
    fig.legend(handles=class_legend(classes, sorted(present)),
               loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout()
    return save_fig(fig, path)


def plot_seasonal_rgb(s2: np.ndarray, dates: list, idx: list[int], patch_id, path=None) -> plt.Figure:
    """RGB composites of one patch at several dates to show seasonal change."""
    fig, axes = plt.subplots(1, len(idx), figsize=(4 * len(idx), 4))
    for ax, t in zip(np.atleast_1d(axes), idx):
        ax.imshow(rgb_composite(s2[t]))
        ax.set_title(pd.Timestamp(dates[t]).strftime("%d %b %Y"))
        ax.axis("off")
    fig.suptitle(f"Patch {patch_id} through the season (true colour)")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_cloud_by_date(cloud: pd.DataFrame, threshold: float, path=None) -> plt.Figure:
    """Share of bright (likely cloud/haze) pixels per acquisition date."""
    fig, ax = plt.subplots(figsize=(13, 4))
    colors = ["tab:red" if v > 0.5 else "tab:orange" if v > 0.1 else "tab:green"
              for v in cloud["bright_frac"]]
    ax.bar(cloud.index, cloud["bright_frac"] * 100, width=5, color=colors)
    ax.set_ylabel(f"% pixels with blue > {threshold}")
    ax.set_title("Proxy for cloud contamination per acquisition date "
                 "(green < 10%, orange 10-50%, red > 50%)")
    ax.set_ylim(0, 100)
    fig.autofmt_xdate()
    fig.tight_layout()
    return save_fig(fig, path)


def plot_ndvi_profiles(ndvi: pd.DataFrame, classes: dict, path=None) -> plt.Figure:
    """Small multiples: mean NDVI over time per class, with background as a grey reference."""
    cmap, _ = pastis_cmap(len(classes))
    crop_ids = [c for c in ndvi.columns if c != 0]
    ncols = 5
    nrows = int(np.ceil(len(crop_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, crop_ids):
        if 0 in ndvi.columns:
            ax.plot(ndvi.index, ndvi[0], color="lightgrey", lw=1, label="Background")
        ax.plot(ndvi.index, ndvi[c], color=cmap(int(c)), lw=2)
        ax.set_title(f"{c}: {classes[int(c)]}", fontsize=10)
        ax.set_ylim(-0.2, 1.0)
        ax.grid(alpha=0.3)
    for ax in axes[len(crop_ids):]:
        ax.axis("off")
    fig.suptitle("Mean NDVI per class over the season (grey = background)", y=1.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    return save_fig(fig, path)


def plot_aoi_map(meta, path=None) -> plt.Figure:
    """Patch footprints in WGS84 coloured by fold, with an optional web basemap."""
    gdf = meta.to_crs(epsg=4326)
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax, column="Fold", categorical=True, cmap="Set1",
             edgecolor="black", linewidth=0.5, alpha=0.6, legend=True,
             legend_kwds={"title": "Fold", "loc": "upper left"})
    try:  # basemap is optional and needs internet access
        import contextily as cx
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik)
    except Exception:
        pass
    minx, miny, maxx, maxy = gdf.total_bounds
    ax.set_title(f"AOI: {len(gdf)} patches\n"
                 f"lon {minx:.2f} to {maxx:.2f}, lat {miny:.2f} to {maxy:.2f}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    return save_fig(fig, path)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def ndvi(s2: np.ndarray) -> np.ndarray:
    """NDVI for every date and pixel: (B8 - B4) / (B8 + B4). Returns (T, H, W)."""
    red = s2[:, 2].astype(np.float32)
    nir = s2[:, 6].astype(np.float32)
    return (nir - red) / (nir + red + 1e-6)


def compute_stats(cfg: dict, meta) -> dict:
    """One pass over all patches to collect class counts, cloud proxy and NDVI profiles."""
    ids = list_patch_ids(cfg["paths"]["s2_dir"], cfg["paths"]["labels_dir"])
    n_classes = len(cfg["classes"])

    dates = meta.loc[ids[0], "dates"]
    if not all(len(meta.loc[p, "dates"]) == len(dates) for p in ids):
        print("[warn] patches have different numbers of dates")
    T = len(dates)

    class_counts = np.zeros((len(ids), n_classes), dtype=np.int64)
    bright_frac = np.zeros((len(ids), T))
    ndvi_sum = np.zeros((n_classes, T))
    ndvi_n = np.zeros(n_classes)

    for i, pid in enumerate(tqdm(ids, desc="Scanning patches")):
        s2, y = load_patch(pid, cfg)
        class_counts[i] = np.bincount(y.ravel(), minlength=n_classes)[:n_classes]
        bright_frac[i] = (s2[:, 0] / REFL_SCALE > BRIGHT_BLUE_THRESHOLD).mean(axis=(1, 2))
        nd = ndvi(s2)
        for c in np.unique(y):
            mask = y == c
            ndvi_sum[c] += nd[:, mask].sum(axis=1)
            ndvi_n[c] += mask.sum()

    counts = pd.DataFrame(class_counts, index=pd.Index(ids, name="patch_id"),
                          columns=range(n_classes))
    cloud = pd.DataFrame({"bright_frac": bright_frac.mean(axis=0)},
                         index=pd.DatetimeIndex(dates, name="date"))
    present = np.where(ndvi_n > 0)[0]
    ndvi_profile = pd.DataFrame(
        {int(c): ndvi_sum[c] / ndvi_n[c] for c in present},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return {"ids": ids, "dates": dates, "counts": counts,
            "cloud": cloud, "ndvi": ndvi_profile}


def class_distribution(counts: pd.DataFrame, classes: dict) -> pd.DataFrame:
    """Pixel totals, pixel share and number of patches containing each class."""
    total = counts.sum()
    dist = pd.DataFrame({
        "class_name": [classes[int(c)] for c in counts.columns],
        "pixels": total.values,
        "pixel_share_pct": (100 * total / total.sum()).values,
        "n_patches": (counts > 0).sum().values,
    }, index=pd.Index(counts.columns, name="class_id"))
    return dist


def class_share_by_fold(counts: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    """Percentage of pixels per class within each fold (each column sums to 100)."""
    by_fold = counts.groupby(folds.loc[counts.index].values).sum().T
    return 100 * by_fold / by_fold.sum()


def pick_diverse_patches(counts: pd.DataFrame, n: int = 6, seed: int = 42) -> list[int]:
    """Choose patches with many crop classes, for more informative example plots."""
    n_crop = (counts.drop(columns=[0, 19], errors="ignore") > 0).sum(axis=1)
    top = n_crop.sort_values(ascending=False).index[: n * 2]
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(top, size=min(n, len(top)), replace=False).tolist())


def seasonal_indices(cloud: pd.DataFrame) -> list[int]:
    """Clearest date in each of autumn, spring, early summer and late summer."""
    windows = [(9, 11), (3, 5), (6, 7), (8, 9)]
    idx = []
    for lo, hi in windows:
        months = cloud.index.month
        cand = np.where((months >= lo) & (months <= hi))[0]
        if len(cand):
            idx.append(int(cand[np.argmin(cloud["bright_frac"].values[cand])]))
    return sorted(set(idx))


def run_eda(cfg: dict, show: bool = False) -> dict:
    """Compute all statistics, save tables and figures, and return the results."""
    classes = {int(k): v for k, v in cfg["classes"].items()}
    fig_dir = Path(cfg["paths"]["figures_dir"])
    met_dir = Path(cfg["paths"]["metrics_dir"])
    met_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(cfg["paths"]["metadata"])
    stats = compute_stats(cfg, meta)
    counts, cloud, nd = stats["counts"], stats["cloud"], stats["ndvi"]

    dist = class_distribution(counts, classes)
    fold_share = class_share_by_fold(counts, meta["Fold"])
    dist.to_csv(met_dir / "class_distribution.csv")
    fold_share.to_csv(met_dir / "class_share_by_fold.csv")
    counts.to_csv(met_dir / "class_pixel_counts_per_patch.csv")
    cloud.to_csv(met_dir / "cloud_proxy_by_date.csv")
    nd.to_csv(met_dir / "ndvi_profile_by_class.csv")

    figs = {}
    figs["class_distribution"] = plot_class_distribution(
        dist, classes, fig_dir / "class_distribution.png")
    figs["class_by_fold"] = plot_class_by_fold(
        fold_share, classes, fig_dir / "class_share_by_fold.png")
    figs["cloud"] = plot_cloud_by_date(
        cloud, BRIGHT_BLUE_THRESHOLD, fig_dir / "cloud_proxy_by_date.png")
    figs["ndvi"] = plot_ndvi_profiles(nd, classes, fig_dir / "ndvi_profiles_by_class.png")
    figs["aoi"] = plot_aoi_map(meta.loc[stats["ids"]], fig_dir / "aoi_map.png")

    examples = pick_diverse_patches(counts, n=6, seed=cfg["seed"])
    labels = {pid: load_patch(pid, cfg)[1] for pid in examples}
    figs["labels"] = plot_label_examples(labels, classes, path=fig_dir / "label_examples.png")

    clear_t = int(np.argmin(cloud["bright_frac"].values))
    clear_date = pd.Timestamp(stats["dates"][clear_t]).strftime("%d %b %Y")
    items = []
    for pid in examples[:3]:
        s2, y = load_patch(pid, cfg)
        items.append((pid, s2[clear_t], y))
    figs["rgb_labels"] = plot_rgb_label_pairs(
        items, classes, clear_date, fig_dir / "rgb_and_labels.png")

    s2, _ = load_patch(examples[0], cfg)
    figs["seasonal"] = plot_seasonal_rgb(
        s2, stats["dates"], seasonal_indices(cloud), examples[0],
        fig_dir / "seasonal_rgb.png")

    if not show:
        plt.close("all")

    print(f"\nPatches: {len(stats['ids'])} | dates: {len(stats['dates'])} "
          f"({stats['dates'][0].date()} to {stats['dates'][-1].date()})")
    print(f"Clearest date: {clear_date} | dates with >50% bright pixels: "
          f"{int((cloud['bright_frac'] > 0.5).sum())}")
    print("\nClass distribution:")
    print(dist.round(3).to_string())
    print(f"\nTables saved to {met_dir}/ and figures to {fig_dir}/")

    stats.update({"meta": meta, "dist": dist, "fold_share": fold_share,
                  "examples": examples, "clear_t": clear_t, "figs": figs})
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_eda(load_config(args.config))


if __name__ == "__main__":
    main()
