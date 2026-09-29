"""Figures of model outputs: predictions, errors, confusion matrices and importance.

Uses the same PASTIS colour map and RGB composite as the EDA (src.eda), so all
figures in the report share one colour scheme. Reads the metrics, prediction
maps and feature importance saved by src.evaluate and src.train.

    python -m src.visualization --config configs/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")  # no display needed when run as a script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.data_loading import load_config, load_patch
from src.eda import class_legend, pastis_cmap, rgb_composite, save_fig
from src.train_test_split import load_split


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_prediction(model: str, patch_id: int, cfg: dict) -> np.ndarray:
    """Saved prediction map for one test patch, in original PASTIS class IDs."""
    return np.load(Path(cfg["paths"]["predictions_dir"]) / model / f"pred_{patch_id}.npy")


def ignored_ids(cfg: dict) -> set[int]:
    """Class IDs that are not scored (void and excluded classes)."""
    return {cfg["void_label"], *cfg.get("excluded_classes", [])}


def pick_rgb_date(cfg: dict, months=(6, 7, 8)) -> tuple[int, str]:
    """Index and label of the clearest summer date, from the EDA cloud table.

    Summer is chosen because most crops are visibly different then; the overall
    clearest date (mid-October) shows mostly bare, harvested fields.
    """
    path = Path(cfg["paths"]["metrics_dir"]) / "cloud_proxy_by_date.csv"
    cloud = pd.read_csv(path, index_col=0, parse_dates=True)
    cand = np.flatnonzero(cloud.index.month.isin(months))
    if len(cand) == 0:
        cand = np.arange(len(cloud))
    t = int(cand[np.argmin(cloud["bright_frac"].values[cand])])
    return t, cloud.index[t].strftime("%d %b %Y")


def patch_accuracies(model: str, ids: list[int], cfg: dict) -> pd.Series:
    """Pixel accuracy per test patch (scored pixels only), used to pick examples."""
    skip = ignored_ids(cfg)
    acc = {}
    for pid in ids:
        _, y = load_patch(pid, cfg)
        pred = load_prediction(model, pid, cfg)
        keep = ~np.isin(y, list(skip))
        acc[pid] = float((pred[keep] == y[keep]).mean()) if keep.any() else np.nan
    return pd.Series(acc).dropna().sort_values(ascending=False)


def select_examples(acc: pd.Series, n: int = 4) -> list[int]:
    """Best, worst and evenly spaced patches in between, ordered best to worst."""
    positions = np.linspace(0, len(acc) - 1, n).round().astype(int)
    return [int(acc.index[i]) for i in dict.fromkeys(positions)]


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_prediction_panels(model: str, ids: list[int], cfg: dict, classes: dict,
                           acc: pd.Series, path=None) -> plt.Figure:
    """Rows of [Sentinel-2 RGB | ground truth | prediction | errors] for test patches."""
    cmap, norm = pastis_cmap(len(classes))
    skip = ignored_ids(cfg)
    t, date_label = pick_rgb_date(cfg)
    err_cmap = ListedColormap(["#2ca02c", "#d62728", "#bdbdbd"])  # correct, wrong, not scored

    fig, axes = plt.subplots(len(ids), 4, figsize=(16, 4 * len(ids)))
    axes = np.atleast_2d(axes)
    present = set()
    for row, pid in zip(axes, ids):
        s2, y = load_patch(pid, cfg)
        pred = load_prediction(model, pid, cfg)
        err = np.where(np.isin(y, list(skip)), 2, (pred != y).astype(int))
        present |= set(np.unique(y).tolist()) | set(np.unique(pred).tolist())

        row[0].imshow(rgb_composite(s2[t]))
        row[0].set_title(f"Patch {pid}: Sentinel-2 RGB ({date_label})")
        row[1].imshow(y, cmap=cmap, norm=norm, interpolation="nearest")
        row[1].set_title("Ground truth")
        row[2].imshow(pred, cmap=cmap, norm=norm, interpolation="nearest")
        row[2].set_title(f"Prediction ({model.replace('_', ' ')})")
        row[3].imshow(err, cmap=err_cmap, vmin=0, vmax=2, interpolation="nearest")
        row[3].set_title(f"Errors: pixel accuracy {acc[pid]:.1%}")
        for ax in row:
            ax.axis("off")

    handles = class_legend(classes, sorted(present))
    handles += [Patch(facecolor="none", edgecolor="none", label=""),
                Patch(facecolor="#2ca02c", label="Correct"),
                Patch(facecolor="#d62728", label="Wrong"),
                Patch(facecolor="#bdbdbd", label="Not scored (void/excluded)")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.suptitle("Test patches from best to worst pixel accuracy", y=1.0, fontsize=14)
    fig.tight_layout()
    return save_fig(fig, path)


def plot_confusion_matrix(model: str, split: str, cfg: dict, path=None) -> plt.Figure:
    """Row-normalised confusion matrix: each row shows where a true class's pixels went."""
    cm = pd.read_csv(Path(cfg["paths"]["metrics_dir"]) / f"confusion_{model}_{split}.csv",
                     index_col=0)
    support = cm.sum(axis=1)
    cm = cm[support > 0]  # drop classes absent from this split
    norm_cm = cm.div(cm.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(norm_cm.values, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(norm_cm.shape[1]), norm_cm.columns, rotation=60, ha="right")
    ax.set_yticks(range(norm_cm.shape[0]),
                  [f"{c} (n={int(support[c]):,})" for c in norm_cm.index])
    for i in range(norm_cm.shape[0]):
        for j in range(norm_cm.shape[1]):
            v = norm_cm.values[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 50 else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class (number of pixels)")
    ax.set_title(f"Confusion matrix, {model.replace('_', ' ')}, {split} set "
                 f"(% of each true class; values < 0.5% hidden)")
    fig.colorbar(im, ax=ax, label="% of true class", shrink=0.7)
    fig.tight_layout()
    return save_fig(fig, path)


def plot_per_class_scores(models: list[str], split: str, cfg: dict, path=None) -> plt.Figure:
    """Per-class F1 and IoU for each model, with the class size shown."""
    met = Path(cfg["paths"]["metrics_dir"])
    tables = {m: pd.read_csv(met / f"per_class_{m}_{split}.csv", index_col=0) for m in models}
    base = tables[models[0]]
    base = base[base["support_pixels"] > 0]
    labels = [f"{n} ({s:,})" for n, s in zip(base["class_name"], base["support_pixels"])]
    y = np.arange(len(base))
    h = 0.8 / len(models)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    for ax, metric in zip(axes, ["f1", "iou"]):
        for k, m in enumerate(models):
            vals = tables[m].loc[base.index, metric].fillna(0)
            ax.barh(y + k * h - 0.4 + h / 2, vals, height=h, label=m.replace("_", " "))
        ax.set_xlim(0, 1)
        ax.set_xlabel(metric.upper() if metric == "iou" else "F1")
        ax.set_title(f"Per-class {metric.upper() if metric == 'iou' else 'F1'} on {split} set")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Class (number of test pixels)")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_model_comparison(cfg: dict, path=None) -> plt.Figure:
    """Overall metrics for every model on validation and test."""
    df = pd.read_csv(Path(cfg["paths"]["metrics_dir"]) / "model_comparison.csv")
    metrics = ["overall_accuracy", "weighted_f1", "macro_f1", "miou", "miou_crops_only"]
    labels = ["Overall\naccuracy", "Weighted\nF1", "Macro\nF1", "mIoU", "mIoU\n(crops)"]
    groups = [(r.model, r.split) for r in df.itertuples()]
    x = np.arange(len(metrics))
    w = 0.8 / len(groups)

    fig, ax = plt.subplots(figsize=(12, 5))
    for k, (m, s) in enumerate(groups):
        vals = df[(df.model == m) & (df.split == s)][metrics].values.ravel()
        bars = ax.bar(x + k * w - 0.4 + w / 2, vals, width=w,
                      label=f"{m.replace('_', ' ')} ({s})",
                      hatch="//" if s == "test" else None, edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on full patches (hatched = test set)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_fig(fig, path)


def plot_feature_importance(model: str, cfg: dict, path=None) -> plt.Figure:
    """Importance summed by channel (band or index) and by acquisition date."""
    met = Path(cfg["paths"]["metrics_dir"])
    by_ch = pd.read_csv(met / f"importance_by_channel_{model}.csv", index_col=0).iloc[:, 0]
    by_date = pd.read_csv(met / f"importance_by_date_{model}.csv", index_col=0).iloc[:, 0]
    by_date.index = pd.to_datetime(by_date.index.astype(str), format="%Y%m%d")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 2]})
    by_ch.sort_values().plot.barh(ax=axes[0], color="tab:blue")
    axes[0].set_xlabel("Share of total importance")
    axes[0].set_title("By band / index (summed over dates)")

    axes[1].bar(by_date.index, by_date.values, width=5, color="tab:green")
    axes[1].set_ylabel("Share of total importance")
    axes[1].set_title("By acquisition date (summed over bands and indices)")
    fig.autofmt_xdate()
    fig.suptitle(f"Feature importance, {model.replace('_', ' ')}", fontsize=13)
    fig.tight_layout()
    return save_fig(fig, path)


def run_visualization(cfg: dict, show: bool = False) -> dict:
    """Create and save all output figures; returns them for display in notebooks."""
    classes = {int(k): v for k, v in cfg["classes"].items()}
    ev = cfg["evaluation"]
    final, models = ev["final_model"], ev["models"]
    out = Path(cfg["paths"]["figures_dir"])
    met = Path(cfg["paths"]["metrics_dir"])

    test_ids = load_split("test", cfg["paths"]["splits_dir"])
    acc = patch_accuracies(final, test_ids, cfg)
    acc.rename("pixel_accuracy").to_csv(met / f"test_patch_accuracy_{final}.csv")
    examples = select_examples(acc, n=4)

    figs = {"predictions": plot_prediction_panels(
        final, examples, cfg, classes, acc, out / f"predictions_{final}_test.png")}
    for m in models:
        for split in ev["splits"]:
            figs[f"cm_{m}_{split}"] = plot_confusion_matrix(
                m, split, cfg, out / f"confusion_matrix_{m}_{split}.png")
        figs[f"importance_{m}"] = plot_feature_importance(
            m, cfg, out / f"feature_importance_{m}.png")
    figs["per_class"] = plot_per_class_scores(models, "test", cfg,
                                              out / "per_class_scores_test.png")
    figs["comparison"] = plot_model_comparison(cfg, out / "model_comparison.png")

    if not show:
        plt.close("all")
    print(f"Test patch accuracy ({final}): mean {acc.mean():.3f}, "
          f"best {acc.max():.3f} (patch {acc.idxmax()}), worst {acc.min():.3f} (patch {acc.idxmin()})")
    print(f"Example patches shown: {examples}")
    print(f"Figures saved to {out}/: " + ", ".join(sorted(p.name for p in out.glob('*.png')
                                                          if not p.name.startswith(('class_', 'aoi', 'label', 'rgb', 'seasonal', 'cloud', 'ndvi')))))
    return figs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model outputs")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_visualization(load_config(args.config))


if __name__ == "__main__":
    main()