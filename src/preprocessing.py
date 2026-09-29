"""Preprocessing and feature engineering for pixel-wise crop classification.

Steps
1. Clip reflectance to [0, 10000] and rescale to [0, 1].
2. Drop acquisition dates that are cloudy across the AOI (decided on training patches only).
3. Add spectral indices (NDVI, NDWI, NDMI, NDRE) for every date.
4. Flatten each pixel's time series into one feature vector (bands and indices x dates).
5. Remap labels to consecutive IDs, ignoring the void label and excluded classes.
6. Sample pixels per class and per patch to build balanced-ish training and
   validation tables.

Build the tables (saved to data/processed/, which is not committed):
    python -m src.preprocessing --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loading import load_config, load_metadata, load_patch
from src.train_test_split import load_split

IGNORE = -1  # label value for pixels that are skipped in training and evaluation
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
B = {name: i for i, name in enumerate(BAND_NAMES)}

# Index definitions: (band a, band b) for the normalised difference (a - b) / (a + b)
INDEX_BANDS = {
    "NDVI": ("B8", "B4"),   # vegetation greenness
    "NDWI": ("B3", "B8"),   # open water / canopy water (McFeeters)
    "NDMI": ("B8A", "B11"),  # leaf and soil moisture
    "NDRE": ("B8A", "B5"),  # red edge, chlorophyll; sensitive later in the season
}


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
def label_mapping(cfg: dict) -> tuple[dict[int, int], dict[int, str]]:
    """Map original PASTIS class IDs to consecutive model IDs.

    Returns (orig_to_model, model_to_name). Void and excluded classes are
    left out of the mapping and become IGNORE.
    """
    classes = {int(k): v for k, v in cfg["classes"].items()}
    skip = {cfg["void_label"], *cfg.get("excluded_classes", [])}
    kept = [c for c in sorted(classes) if c not in skip]
    orig_to_model = {c: i for i, c in enumerate(kept)}
    model_to_name = {i: classes[c] for c, i in orig_to_model.items()}
    return orig_to_model, model_to_name


def remap_labels(y: np.ndarray, orig_to_model: dict[int, int]) -> np.ndarray:
    """Convert an original label map to model IDs, with IGNORE for skipped classes."""
    lut = np.full(256, IGNORE, dtype=np.int64)
    for orig, new in orig_to_model.items():
        lut[orig] = new
    return lut[y]


# --------------------------------------------------------------------------
# Imagery
# --------------------------------------------------------------------------
def to_reflectance(s2: np.ndarray, reflectance_max: float = 10000) -> np.ndarray:
    """Clip raw int16 values to [0, max] and rescale to float32 reflectance in [0, 1]."""
    return np.clip(s2.astype(np.float32), 0, reflectance_max) / reflectance_max


def cloudy_date_mask(cfg: dict, patch_ids: list[int]) -> np.ndarray:
    """Boolean mask over dates: True = keep.

    A date is dropped if, averaged over the given patches, more than
    `cloud_date_max_frac` of pixels have blue reflectance above
    `cloud_blue_threshold`. All patches share one tile and one date axis, so a
    single mask applies to every patch. Pass training patches only.
    """
    pp = cfg["preprocessing"]
    fracs = []
    for pid in patch_ids:
        s2 = np.load(Path(cfg["paths"]["s2_dir"]) / f"S2_{pid}.npy", mmap_mode="r")
        blue = np.asarray(s2[:, B["B2"]], dtype=np.float32) / pp["reflectance_max"]
        fracs.append((blue > pp["cloud_blue_threshold"]).mean(axis=(1, 2)))
    frac = np.mean(fracs, axis=0)
    return frac <= pp["cloud_date_max_frac"]


def add_indices(refl: np.ndarray, indices: list[str]) -> np.ndarray:
    """Normalised-difference indices for every date. Input (T, 10, H, W) -> (T, n_idx, H, W)."""
    out = []
    for name in indices:
        a, b = INDEX_BANDS[name]
        va, vb = refl[:, B[a]], refl[:, B[b]]
        out.append((va - vb) / (va + vb + 1e-6))
    return np.stack(out, axis=1).astype(np.float32)


def feature_names(dates: list, indices: list[str]) -> list[str]:
    """Column names in the same order as `patch_features` produces them."""
    labels = [pd.Timestamp(d).strftime("%Y%m%d") for d in dates]
    channels = BAND_NAMES + list(indices)
    return [f"{ch}_{d}" for d in labels for ch in channels]


def patch_features(s2: np.ndarray, keep_dates: np.ndarray, cfg: dict) -> np.ndarray:
    """Full feature matrix for one patch: (H * W, n_dates * (10 + n_indices)).

    Row order is row-major over the patch, so predictions reshape back to (H, W).
    """
    pp = cfg["preprocessing"]
    refl = to_reflectance(s2[keep_dates], pp["reflectance_max"])
    stack = np.concatenate([refl, add_indices(refl, pp["indices"])], axis=1)  # (T, C, H, W)
    T, C, H, W = stack.shape
    return stack.transpose(2, 3, 0, 1).reshape(H * W, T * C)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def sample_pixels(y: np.ndarray, max_per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Flat indices of up to `max_per_class` random pixels per class (IGNORE skipped)."""
    flat = y.ravel()
    picked = []
    for c in np.unique(flat):
        if c == IGNORE:
            continue
        idx = np.flatnonzero(flat == c)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        picked.append(idx)
    return np.sort(np.concatenate(picked)) if picked else np.array([], dtype=np.int64)


def build_table(cfg: dict, patch_ids: list[int], keep_dates: np.ndarray,
                orig_to_model: dict, seed: int, desc: str = "") -> tuple:
    """Sampled feature table for a set of patches: X, y and the source patch of each row."""
    rng = np.random.default_rng(seed)
    cap = cfg["sampling"]["max_pixels_per_class_per_patch"]
    Xs, ys, ps = [], [], []
    for pid in tqdm(patch_ids, desc=desc):
        s2, y_orig = load_patch(pid, cfg)
        y = remap_labels(y_orig, orig_to_model)
        idx = sample_pixels(y, cap, rng)
        if len(idx) == 0:
            continue
        Xs.append(patch_features(s2, keep_dates, cfg)[idx])
        ys.append(y.ravel()[idx])
        ps.append(np.full(len(idx), pid))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pixel feature tables")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits_dir = cfg["paths"]["splits_dir"]
    train_ids, val_ids = load_split("train", splits_dir), load_split("val", splits_dir)
    orig_to_model, model_to_name = label_mapping(cfg)

    meta = load_metadata(cfg["paths"]["metadata"])
    dates = list(meta.loc[train_ids[0], "dates"])
    if cfg["preprocessing"]["drop_cloudy_dates"]:
        keep = cloudy_date_mask(cfg, train_ids)
    else:
        keep = np.ones(len(dates), dtype=bool)
    kept_dates = [d for d, k in zip(dates, keep) if k]
    dropped = [pd.Timestamp(d).date().isoformat() for d, k in zip(dates, keep) if not k]
    names = feature_names(kept_dates, cfg["preprocessing"]["indices"])
    print(f"Dates kept: {keep.sum()} of {len(keep)} | dropped as cloudy: {dropped}")
    print(f"Features per pixel: {len(names)} "
          f"({keep.sum()} dates x {10 + len(cfg['preprocessing']['indices'])} channels)")
    print(f"Model classes ({len(model_to_name)}): {model_to_name}")

    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, ids, seed in [("train", train_ids, cfg["seed"]), ("val", val_ids, cfg["seed"] + 1)]:
        X, y, p = build_table(cfg, ids, keep, orig_to_model, seed, desc=f"Building {name}")
        np.savez_compressed(out_dir / f"{name}.npz", X=X, y=y, patch_id=p)
        summary[name] = np.bincount(y, minlength=len(model_to_name))
        print(f"{name}: X {X.shape}, {X.nbytes / 1e6:.0f} MB in memory")

    # Everything later steps need to reproduce the features exactly
    prep_info = {
        "keep_dates": keep.tolist(),
        "kept_dates": [pd.Timestamp(d).date().isoformat() for d in kept_dates],
        "dropped_dates": dropped,
        "feature_names": names,
        "orig_to_model": {str(k): v for k, v in orig_to_model.items()},
        "model_to_name": {str(k): v for k, v in model_to_name.items()},
    }
    (out_dir / "preprocessing_info.json").write_text(json.dumps(prep_info, indent=2))
    met_dir = Path(cfg["paths"]["metrics_dir"])
    met_dir.mkdir(parents=True, exist_ok=True)
    (met_dir / "preprocessing_info.json").write_text(json.dumps(prep_info, indent=2))

    table = pd.DataFrame({"class_name": list(model_to_name.values()),
                          "train_samples": summary["train"],
                          "val_samples": summary["val"]},
                         index=pd.Index(list(model_to_name), name="model_id"))
    table.to_csv(met_dir / "sampled_class_counts.csv")
    print("\nSampled pixels per class:")
    print(table.to_string())
    print(f"\nTables saved to {out_dir}/, summary to {met_dir}/")


if __name__ == "__main__":
    main()