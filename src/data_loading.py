"""Data loading utilities for the PASTIS-derived crop classification dataset.

Expected layout (paths set in configs/config.yaml):
    data/DATA_S2/S2_<patch_id>.npy          -> (T, 10, H, W) Sentinel-2 time series
    data/ANNOTATIONS/TARGET_<patch_id>.npy  -> (1, H, W) semantic labels (layer 0)
    data/metadata.geojson                   -> patch footprints and acquisition dates

Run directly for a quick inspection of the dataset:
    python -m src.data_loading --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    """Read the YAML config into a dictionary."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Config file '{path}' is empty or could not be parsed.")
    return cfg
    


def list_patch_ids(s2_dir: str | Path, labels_dir: str | Path) -> list[int]:
    """Return sorted patch IDs that have both an S2 file and a label file.

    Warns about any patch that exists in only one of the two folders.
    """
    s2_ids = {int(p.stem.split("_")[1]) for p in Path(s2_dir).glob("S2_*.npy")}
    lbl_ids = {int(p.stem.split("_")[1]) for p in Path(labels_dir).glob("TARGET_*.npy")}

    missing_labels = s2_ids - lbl_ids
    missing_images = lbl_ids - s2_ids
    if missing_labels:
        print(f"[warn] {len(missing_labels)} patches have imagery but no labels")
    if missing_images:
        print(f"[warn] {len(missing_images)} patches have labels but no imagery")

    return sorted(s2_ids & lbl_ids)


def load_s2(patch_id: int, s2_dir: str | Path) -> np.ndarray:
    """Load a Sentinel-2 time series of shape (T, 10, H, W)."""
    return np.load(Path(s2_dir) / f"S2_{patch_id}.npy")


def load_target(patch_id: int, labels_dir: str | Path) -> np.ndarray:
    """Load the semantic label map and return it as (H, W)."""
    target = np.load(Path(labels_dir) / f"TARGET_{patch_id}.npy")
    if target.ndim == 3:
        target = target[0]
    return target.astype(np.int64)


def _parse_dates(value) -> list[pd.Timestamp]:
    """Parse the 'dates-S2' field, which may be a dict or a string of a dict.

    In PASTIS the field maps observation index -> date as YYYYMMDD.
    Dates are returned in observation-index order so they align with axis 0
    of the S2 array.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    items = sorted(value.items(), key=lambda kv: int(kv[0]))
    return [pd.to_datetime(str(v), format="%Y%m%d") for _, v in items]


def load_metadata(path: str | Path):
    """Load metadata.geojson as a GeoDataFrame indexed by patch ID.

    Adds a 'dates' column with parsed acquisition dates when available.
    """
    import geopandas as gpd  # imported here so the rest works without geopandas

    gdf = gpd.read_file(path)
    id_col = "ID_PATCH" if "ID_PATCH" in gdf.columns else gdf.columns[0]
    gdf[id_col] = gdf[id_col].astype(int)
    gdf = gdf.set_index(id_col)

    if "dates-S2" in gdf.columns:
        gdf["dates"] = gdf["dates-S2"].apply(_parse_dates)
    return gdf


def load_patch(patch_id: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning (s2, target) for one patch."""
    s2 = load_s2(patch_id, cfg["paths"]["s2_dir"])
    target = load_target(patch_id, cfg["paths"]["labels_dir"])
    if s2.shape[-2:] != target.shape:
        raise ValueError(
            f"Patch {patch_id}: image {s2.shape[-2:]} and label {target.shape} differ"
        )
    return s2, target


def inspect_dataset(cfg: dict) -> pd.DataFrame:
    """Print and return a per-patch summary of shapes, dtypes and value ranges."""
    ids = list_patch_ids(cfg["paths"]["s2_dir"], cfg["paths"]["labels_dir"])
    print(f"Patches with imagery and labels: {len(ids)}")

    rows = []
    for pid in ids:
        s2, target = load_patch(pid, cfg)
        rows.append(
            {
                "patch_id": pid,
                "n_obs": s2.shape[0],
                "n_bands": s2.shape[1],
                "height": s2.shape[2],
                "width": s2.shape[3],
                "s2_dtype": str(s2.dtype),
                "s2_min": float(np.nanmin(s2)),
                "s2_max": float(np.nanmax(s2)),
                "s2_nan_frac": float(np.isnan(s2).mean()) if s2.dtype.kind == "f" else 0.0,
                "label_dtype": str(np.load(
                    Path(cfg["paths"]["labels_dir"]) / f"TARGET_{pid}.npy"
                ).dtype),
                "n_classes_in_patch": len(np.unique(target)),
            }
        )
    summary = pd.DataFrame(rows).set_index("patch_id")

    print("\nShapes and dtypes")
    print(f"  bands per obs      : {sorted(int(b) for b in summary.n_bands.unique())}")
    print(f"  patch size (HxW)   : {sorted({(int(h), int(w)) for h, w in zip(summary.height, summary.width)})}")
    print(f"  S2 dtype(s)        : {sorted(summary.s2_dtype.unique())}")
    print(f"  label dtype(s)     : {sorted(summary.label_dtype.unique())}")
    print("\nObservations per patch (T)")
    print(summary.n_obs.describe().round(1).to_string())
    print("\nReflectance value range")
    print(f"  min {summary.s2_min.min():.1f} | max {summary.s2_max.max():.1f}")
    print(f"  max NaN fraction in any patch: {summary.s2_nan_frac.max():.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the crop classification dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = inspect_dataset(cfg)

    out = Path(cfg["paths"]["metrics_dir"]) / "patch_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out)
    print(f"\nSaved per-patch summary to {out}")

    meta_path = Path(cfg["paths"]["metadata"])
    if meta_path.exists():
        meta = load_metadata(meta_path)
        print(f"\nMetadata: {len(meta)} rows, CRS = {meta.crs}")
        print(f"  columns: {list(meta.columns)}")
        if "dates" in meta.columns:
            first = meta.index.intersection(summary.index)[0]
            d = meta.loc[first, "dates"]
            print(f"  patch {first}: {len(d)} dates, {d[0].date()} -> {d[-1].date()}"
                  f" (S2 array has {summary.loc[first, 'n_obs']} obs)")


if __name__ == "__main__":
    main()