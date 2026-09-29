"""Train pixel-wise crop classifiers on the sampled feature tables.

Two models are trained on the same features:
- Random Forest: a simple, robust baseline.
- LightGBM: gradient-boosted trees, usually stronger on tabular data, with
  early stopping on the validation set.

Class imbalance is handled with sample weights (see `class_weights`).
Models are saved to models/ (not committed); a training summary with settings,
timings and validation scores is saved to outputs/metrics/.

    python -m src.train --config configs/config.yaml
    python -m src.train --config configs/config.yaml --models lightgbm
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data_loading import load_config


def load_table(name: str, processed_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a sampled feature table (X, y) built by src.preprocessing."""
    data = np.load(Path(processed_dir) / f"{name}.npz")
    return data["X"], data["y"]


def class_weights(y: np.ndarray, n_classes: int, method: str = "sqrt_inverse") -> np.ndarray:
    """Per-class weights, normalised so the average sample weight is 1.

    - none:          every sample weighs 1.
    - balanced:      weight proportional to 1 / class frequency. Fully evens out
                     classes, but hugely up-weights classes with very few samples.
    - sqrt_inverse:  weight proportional to 1 / sqrt(class frequency). A softer
                     compromise that boosts rare classes without letting a few
                     fields dominate training.
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.zeros(n_classes)
    present = counts > 0
    if method == "none":
        w[present] = 1.0
    elif method == "balanced":
        w[present] = 1.0 / counts[present]
    elif method == "sqrt_inverse":
        w[present] = 1.0 / np.sqrt(counts[present])
    else:
        raise ValueError(f"Unknown class_weighting '{method}'")
    w *= len(y) / (w[y]).sum()  # mean sample weight = 1
    return w


def val_scores(model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Quick validation scores; macro F1 only over classes present in validation."""
    pred = model.predict(X_val)
    present = np.unique(y_val)
    return {
        "accuracy": float(accuracy_score(y_val, pred)),
        "macro_f1": float(f1_score(y_val, pred, labels=present, average="macro", zero_division=0)),
        "n_classes_evaluated": int(len(present)),
    }


def train_random_forest(X, y, sample_weight, params: dict, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(random_state=seed, **params)
    model.fit(X, y, sample_weight=sample_weight)
    return model, {}


def train_lightgbm(X, y, sample_weight, X_val, y_val, params: dict, seed: int):
    import lightgbm as lgb

    # eval_set is deprecated in the newest LightGBM releases but still works, and
    # keeps the script compatible with older versions.
    warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
    params = dict(params)
    stop_rounds = params.pop("early_stopping_rounds")
    model = lgb.LGBMClassifier(objective="multiclass", random_state=seed,
                               verbose=-1, **params)
    model.fit(
        X, y, sample_weight=sample_weight,
        eval_set=[(X_val, y_val)], eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stop_rounds, verbose=False), lgb.log_evaluation(50)],
    )
    return model, {"best_iteration": int(model.best_iteration_ or params["n_estimators"])}


def feature_importance(model, names: list[str], n_channels: int) -> dict[str, pd.DataFrame]:
    """Feature importance per feature, and summed per channel and per date."""
    imp = pd.Series(model.feature_importances_, index=names, dtype=float)
    imp = imp / imp.sum()
    parts = imp.index.str.rsplit("_", n=1, expand=True)
    df = pd.DataFrame({"feature": imp.index, "channel": parts.get_level_values(0),
                       "date": parts.get_level_values(1), "importance": imp.values})
    return {
        "features": df.sort_values("importance", ascending=False),
        "by_channel": df.groupby("channel")["importance"].sum().sort_values(ascending=False),
        "by_date": df.groupby("date")["importance"].sum().sort_index(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop classification models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models", nargs="+", default=None,
                        help="override config, e.g. --models lightgbm")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    proc_dir = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    met_dir = Path(cfg["paths"]["metrics_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((proc_dir / "preprocessing_info.json").read_text())
    names = info["feature_names"]
    n_classes = len(info["model_to_name"])
    n_channels = 10 + len(cfg["preprocessing"]["indices"])

    X, y = load_table("train", proc_dir)
    X_val, y_val = load_table("val", proc_dir)
    print(f"Train {X.shape}, validation {X_val.shape}, {n_classes} classes")

    method = cfg["training"]["class_weighting"]
    cw = class_weights(y, n_classes, method)
    sw = cw[y]
    print(f"Class weights ({method}): " + ", ".join(f"{i}:{w:.2f}" for i, w in enumerate(cw)))

    summary = {
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "cpu_count": os.cpu_count()},
        "data": {"n_train": int(len(y)), "n_val": int(len(y_val)),
                 "n_features": int(X.shape[1]), "n_classes": n_classes},
        "class_weighting": method,
        "class_weights": {info["model_to_name"][str(i)]: round(float(w), 4)
                          for i, w in enumerate(cw)},
        "models": {},
    }

    for name in args.models or cfg["training"]["models"]:
        print(f"\n=== Training {name} ===")
        t0 = time.time()
        if name == "random_forest":
            model, extra = train_random_forest(X, y, sw, cfg["random_forest"], seed)
        elif name == "lightgbm":
            model, extra = train_lightgbm(X, y, sw, X_val, y_val, cfg["lightgbm"], seed)
        else:
            raise ValueError(f"Unknown model '{name}'")
        train_time = time.time() - t0

        scores = val_scores(model, X_val, y_val)
        path = models_dir / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        size_mb = path.stat().st_size / 1e6

        imp = feature_importance(model, names, n_channels)
        imp["features"].to_csv(met_dir / f"feature_importance_{name}.csv", index=False)
        imp["by_channel"].to_csv(met_dir / f"importance_by_channel_{name}.csv")
        imp["by_date"].to_csv(met_dir / f"importance_by_date_{name}.csv")

        summary["models"][name] = {"params": cfg[name], "train_seconds": round(train_time, 1),
                                   "model_file": str(path), "model_size_mb": round(size_mb, 1),
                                   "validation": scores, **extra}
        print(f"Trained in {train_time:.0f}s | saved {path} ({size_mb:.0f} MB)")
        print(f"Validation accuracy {scores['accuracy']:.3f} | "
              f"macro F1 {scores['macro_f1']:.3f} over {scores['n_classes_evaluated']} classes")
        print("Top channels by importance: "
              + ", ".join(f"{c} {v:.2f}" for c, v in imp["by_channel"].head(5).items()))

    (met_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nTraining summary saved to {met_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()