"""Patch-level train / validation / test split using the PASTIS spatial folds.

Splitting is done by whole patch and by fold, never by pixel: neighbouring pixels
of the same field are near-duplicates, so a pixel-level split would leak
information and inflate scores. Folds are assigned in the config.

Create the split, save patch IDs to text files and report class coverage:
    python -m src.split --config configs/config.yaml

Also compare every possible choice of validation and test fold:
    python -m src.split --config configs/config.yaml --compare
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import list_patch_ids, load_config, load_metadata, load_target

SPLITS = ("train", "val", "test")


def class_counts_per_patch(cfg: dict, ids: list[int]) -> pd.DataFrame:
    """Pixel count of every class in every patch (labels only, so this is fast)."""
    n_classes = len(cfg["classes"])
    rows = [np.bincount(load_target(pid, cfg["paths"]["labels_dir"]).ravel(),
                        minlength=n_classes)[:n_classes] for pid in ids]
    return pd.DataFrame(rows, index=pd.Index(ids, name="patch_id"), columns=range(n_classes))


def assign_splits(folds: pd.Series, train: list, val: list, test: list) -> dict[str, list[int]]:
    """Map each split name to the sorted patch IDs in its folds."""
    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    if overlap:
        raise ValueError(f"Folds {sorted(overlap)} are assigned to more than one split")
    out = {name: sorted(folds[folds.isin(f)].index.tolist())
           for name, f in zip(SPLITS, (train, val, test))}
    unused = set(folds.index) - set().union(*out.values())
    if unused:
        print(f"[warn] {len(unused)} patches are not in any split")
    return out


def split_class_table(counts: pd.DataFrame, splits: dict, classes: dict) -> pd.DataFrame:
    """Pixels and patch counts per class for each split."""
    table = pd.DataFrame({"class_name": [classes[c] for c in counts.columns]},
                         index=pd.Index(counts.columns, name="class_id"))
    for name, ids in splits.items():
        sub = counts.loc[ids]
        table[f"{name}_pixels"] = sub.sum().values
        table[f"{name}_patches"] = (sub > 0).sum().values
    return table


def missing_crops(table: pd.DataFrame, crop_ids: list[int]) -> dict[str, list[int]]:
    """Crop classes with zero pixels in each split."""
    return {s: [c for c in crop_ids if table.loc[c, f"{s}_pixels"] == 0] for s in SPLITS}


def compare_fold_choices(counts, folds, crop_ids, classes) -> pd.DataFrame:
    """Score every (val fold, test fold) choice, training on the remaining folds."""
    all_folds = sorted(folds.unique())
    rows = []
    for val_f, test_f in permutations(all_folds, 2):
        train_f = [f for f in all_folds if f not in (val_f, test_f)]
        splits = assign_splits(folds, train_f, [val_f], [test_f])
        miss = missing_crops(split_class_table(counts, splits, classes), crop_ids)
        rows.append({
            "train_folds": ",".join(map(str, train_f)), "val_fold": val_f, "test_fold": test_f,
            "n_train": len(splits["train"]), "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "missing_train": len(miss["train"]), "missing_val": len(miss["val"]),
            "missing_test": len(miss["test"]),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["missing_train", "missing_test", "missing_val"]).reset_index(drop=True)


def save_splits(splits: dict, out_dir: str | Path) -> None:
    """Write one patch ID per line to train_ids.txt, val_ids.txt and test_ids.txt."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out_dir / f"{name}_ids.txt").write_text("\n".join(map(str, ids)) + "\n")


def load_split(name: str, splits_dir: str | Path) -> list[int]:
    """Read patch IDs for one split back from its text file."""
    return [int(x) for x in (Path(splits_dir) / f"{name}_ids.txt").read_text().split()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the fold-based data split")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--compare", action="store_true",
                        help="compare all choices of validation and test fold")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = {int(k): v for k, v in cfg["classes"].items()}
    crop_ids = [c for c in classes if c not in (0, cfg["void_label"])]

    ids = list_patch_ids(cfg["paths"]["s2_dir"], cfg["paths"]["labels_dir"])
    folds = load_metadata(cfg["paths"]["metadata"]).loc[ids, "Fold"].astype(int)
    counts = class_counts_per_patch(cfg, ids)
    met_dir = Path(cfg["paths"]["metrics_dir"])
    met_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        comp = compare_fold_choices(counts, folds, crop_ids, classes)
        comp.to_csv(met_dir / "split_fold_comparison.csv", index=False)
        print("Crop classes missing from each split, for every val/test fold choice")
        print("(best first; training should never miss a class):\n")
        print(comp.to_string(index=False))
        print()

    sc = cfg["split"]
    splits = assign_splits(folds, sc["train_folds"], sc["val_folds"], sc["test_folds"])
    save_splits(splits, cfg["paths"]["splits_dir"])
    table = split_class_table(counts, splits, classes)
    table.to_csv(met_dir / "split_class_coverage.csv")

    print(f"Configured split: train folds {sc['train_folds']}, "
          f"val folds {sc['val_folds']}, test folds {sc['test_folds']}")
    for name in SPLITS:
        print(f"  {name:5s}: {len(splits[name]):3d} patches")
    print("\nPixels per class in each split:")
    print(table[["class_name"] + [f"{s}_pixels" for s in SPLITS]].to_string())
    for name, miss in missing_crops(table, crop_ids).items():
        if miss:
            print(f"\n[note] crops missing from {name}: "
                  + ", ".join(f"{c} ({classes[c]})" for c in miss))
    print(f"\nSplit files saved to {cfg['paths']['splits_dir']}/")


if __name__ == "__main__":
    main()