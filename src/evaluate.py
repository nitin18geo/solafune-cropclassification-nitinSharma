"""Evaluate trained models on every pixel of the validation and test patches.

Training used sampled pixels; evaluation predicts complete 128 x 128 patches,
so the metrics reflect real class frequencies. Void and excluded classes are
predicted but ignored when scoring.

Outputs (outputs/metrics/):
    metrics_summary.json                 overall scores for every model and split
    per_class_<model>_<split>.csv        precision, recall, F1, IoU and support per class
    confusion_<model>_<split>.csv        confusion matrix (rows = true, columns = predicted)
    top_confusions_<model>_<split>.csv   most frequent class-to-class errors
Prediction maps (outputs/predictions/<model>/pred_<patch_id>.npy) hold original
PASTIS class IDs, so they can be compared directly with the label files.

    python -m src.evaluate --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loading import load_config, load_patch
from src.preprocessing import IGNORE, patch_features, remap_labels
from src.train_test_split import load_split

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


def load_prep_info(cfg: dict) -> dict:
    """Preprocessing settings saved by src.preprocessing (dates kept, label mapping)."""
    path = Path(cfg["paths"]["processed_dir"]) / "preprocessing_info.json"
    if not path.exists():
        path = Path(cfg["paths"]["metrics_dir"]) / "preprocessing_info.json"
    info = json.loads(path.read_text())
    info["keep_dates"] = np.array(info["keep_dates"], dtype=bool)
    info["orig_to_model"] = {int(k): v for k, v in info["orig_to_model"].items()}
    info["model_to_name"] = {int(k): v for k, v in info["model_to_name"].items()}
    return info


def predict_patch(model, s2: np.ndarray, info: dict, cfg: dict) -> np.ndarray:
    """Predict every pixel of a patch; returns an (H, W) map of model class IDs."""
    X = patch_features(s2, info["keep_dates"], cfg)
    return model.predict(X).astype(np.int64).reshape(s2.shape[-2:])


def to_original_ids(pred: np.ndarray, info: dict) -> np.ndarray:
    """Convert model class IDs back to original PASTIS class IDs."""
    lut = np.zeros(len(info["model_to_name"]), dtype=np.uint8)
    for orig, new in info["orig_to_model"].items():
        lut[new] = orig
    return lut[pred]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    """Confusion matrix counts (rows = true, columns = predicted), ignoring IGNORE pixels."""
    keep = y_true != IGNORE
    return np.bincount(n * y_true[keep] + y_pred[keep], minlength=n * n).reshape(n, n)


def metrics_from_confusion(cm: np.ndarray, names: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    """Per-class and overall metrics from a confusion matrix.

    Averages (macro F1, mIoU) use only classes present in the evaluated split,
    because a class with no true pixels has no meaningful recall or IoU.
    """
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(precision + np.nan_to_num(recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
        iou = tp / (support + predicted - tp)
    present = support > 0
    f1 = np.where(present, f1, np.nan)
    iou = np.where(present, iou, np.nan)

    per_class = pd.DataFrame({
        "class_name": [names[i] for i in range(len(names))],
        "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        "support_pixels": support.astype(int), "predicted_pixels": predicted.astype(int),
    }, index=pd.Index(range(len(names)), name="model_id"))

    crops = present.copy()
    crops[0] = False  # background is class 0 in the model mapping
    overall = {
        "overall_accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(np.nanmean(f1[present])),
        "weighted_f1": float(np.nansum(f1 * support) / support.sum()),
        "miou": float(np.nanmean(iou[present])),
        "miou_crops_only": float(np.nanmean(iou[crops])),
        "n_classes_evaluated": int(present.sum()),
        "classes_not_in_split": [names[i] for i in np.flatnonzero(~present)],
        "n_pixels_evaluated": int(cm.sum()),
    }
    return per_class, overall


def top_confusions(cm: np.ndarray, names: dict[int, str], k: int = 15) -> pd.DataFrame:
    """Largest off-diagonal errors, with the share of the true class they represent."""
    rows = []
    support = cm.sum(axis=1)
    for t in range(len(cm)):
        for p in range(len(cm)):
            if t != p and cm[t, p] > 0:
                rows.append({"true_class": names[t], "predicted_as": names[p],
                             "pixels": int(cm[t, p]),
                             "share_of_true_class_pct": round(100 * cm[t, p] / support[t], 1)})
    df = pd.DataFrame(rows)
    return df.sort_values("pixels", ascending=False).head(k).reset_index(drop=True) if len(df) else df


def evaluate_model(model_name: str, cfg: dict, info: dict) -> dict:
    """Evaluate one saved model on the configured splits; save metrics and predictions."""
    ev = cfg["evaluation"]
    met_dir = Path(cfg["paths"]["metrics_dir"])
    model = joblib.load(Path(cfg["paths"]["models_dir"]) / f"{model_name}.joblib")
    names = info["model_to_name"]
    n = len(names)
    results = {}

    for split in ev["splits"]:
        ids = load_split(split, cfg["paths"]["splits_dir"])
        save_preds = split in ev.get("save_predictions_for", [])
        pred_dir = Path(cfg["paths"]["predictions_dir"]) / model_name
        if save_preds:
            pred_dir.mkdir(parents=True, exist_ok=True)

        cm = np.zeros((n, n), dtype=np.int64)
        t0 = time.time()
        for pid in tqdm(ids, desc=f"{model_name} / {split}"):
            s2, y_orig = load_patch(pid, cfg)
            pred = predict_patch(model, s2, info, cfg)
            y = remap_labels(y_orig, info["orig_to_model"])
            cm += confusion_matrix(y.ravel(), pred.ravel(), n)
            if save_preds:
                np.save(pred_dir / f"pred_{pid}.npy", to_original_ids(pred, info))
        seconds = time.time() - t0

        per_class, overall = metrics_from_confusion(cm, names)
        overall["n_patches"] = len(ids)
        overall["inference_seconds"] = round(seconds, 1)
        per_class.round(4).to_csv(met_dir / f"per_class_{model_name}_{split}.csv")
        pd.DataFrame(cm, index=[names[i] for i in range(n)],
                     columns=[names[i] for i in range(n)]).to_csv(
            met_dir / f"confusion_{model_name}_{split}.csv")
        top = top_confusions(cm, names)
        top.to_csv(met_dir / f"top_confusions_{model_name}_{split}.csv", index=False)
        results[split] = overall

        print(f"\n{model_name} on {split} ({len(ids)} patches, "
              f"{overall['n_pixels_evaluated']:,} pixels, {seconds:.0f}s)")
        print(f"  OA {overall['overall_accuracy']:.3f} | macro F1 {overall['macro_f1']:.3f} | "
              f"weighted F1 {overall['weighted_f1']:.3f} | mIoU {overall['miou']:.3f} | "
              f"mIoU crops {overall['miou_crops_only']:.3f}")
        if overall["classes_not_in_split"]:
            print(f"  not in {split}: {', '.join(overall['classes_not_in_split'])}")
        if split == "test":
            print(per_class[["class_name", "precision", "recall", "f1", "iou",
                             "support_pixels"]].round(3).to_string())
            print("\n  Top confusions:")
            print(top.head(8).to_string(index=False))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on full patches")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    info = load_prep_info(cfg)
    met_dir = Path(cfg["paths"]["metrics_dir"])
    met_dir.mkdir(parents=True, exist_ok=True)

    summary_path = met_dir / "metrics_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in args.models or cfg["evaluation"]["models"]:
        summary[name] = evaluate_model(name, cfg, info)
    summary["final_model"] = cfg["evaluation"]["final_model"]
    summary_path.write_text(json.dumps(summary, indent=2))

    rows = [{"model": m, "split": s, **{k: round(v, 3) for k, v in r.items()
                                        if k in ("overall_accuracy", "macro_f1",
                                                 "weighted_f1", "miou", "miou_crops_only")}}
            for m, splits in summary.items() if isinstance(splits, dict)
            for s, r in splits.items()]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(met_dir / "model_comparison.csv", index=False)
    print("\nModel comparison (full patches):")
    print(comparison.to_string(index=False))
    print(f"\nMetrics saved to {met_dir}/, predictions to {cfg['paths']['predictions_dir']}/")


if __name__ == "__main__":
    main()