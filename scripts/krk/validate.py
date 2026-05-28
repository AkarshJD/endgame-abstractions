"""
KRK WDL classifier validation.

Runs the trained classifier over the full KRK dataset and reports:
  - Full-dataset accuracy
  - Misclassified positions (FEN + true / predicted WDL)

Usage:
    python scripts/krk/validate.py
"""

import json
import joblib
import pandas as pd
from pathlib import Path

from endgame.utils.logger import make_run_dir, save_json


FEATURES_PATH = "data/processed/krk/features.csv"
RAW_PATH = "data/raw/krk/full.csv"
MODEL_BASE = "logs/krk_classifier"


def find_latest_model(base=MODEL_BASE):
    runs = sorted(Path(base).glob("*/model.joblib"))
    if not runs:
        raise FileNotFoundError(f"No model.joblib found under {base}")
    return runs[-1]


def main():
    model_path = find_latest_model()
    print(f"Model: {model_path}")

    model = joblib.load(model_path)

    meta_path = Path(model_path).parent / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    print("Loading features...")
    features_df = pd.read_csv(FEATURES_PATH)

    if "wdl" not in features_df.columns:
        raise ValueError(
            "No 'wdl' column. Re-run build_dataset.py + build_features.py."
        )

    X = features_df.reindex(columns=feature_cols)
    y_true = features_df["wdl"]

    print("Loading raw dataset (with FEN)...")
    raw_df = pd.read_csv(RAW_PATH)

    print(f"Predicting {len(X):,} positions...")
    y_pred = model.predict(X)

    features_df["prediction"] = y_pred
    features_df["correct"] = y_pred == y_true.values

    accuracy = features_df["correct"].mean()
    misclassified_idx = features_df[~features_df["correct"]].index

    print(f"\nFull-dataset accuracy : {accuracy:.6f}")
    print(f"Total positions       : {len(features_df):,}")
    print(f"Misclassified         : {len(misclassified_idx):,}")

    print("\n=== Class breakdown ===")
    for wdl_val in sorted(y_true.unique()):
        mask = y_true == wdl_val
        acc = (y_pred[mask] == wdl_val).mean()
        label = {2: "WIN", 0: "DRAW", -2: "LOSS"}.get(wdl_val, str(wdl_val))
        print(f"  {label:5s}: {acc:.6f}  ({mask.sum():,} positions)")

    run_dir = make_run_dir(base="logs/krk_validation")

    features_df.to_csv(f"{run_dir}/full_predictions.csv", index=False)

    fens = []
    if len(raw_df) == len(features_df):
        misclassified_raw = raw_df.loc[misclassified_idx]
        misclassified_raw.to_csv(f"{run_dir}/misclassified_full.csv", index=False)

        print("\n===== MISCLASSIFIED FEN POSITIONS =====\n")
        for idx in misclassified_idx[:50]:
            fen = raw_df.loc[idx, "fen"]
            true_label = int(y_true.iloc[idx])
            pred_label = int(y_pred[idx])
            fens.append(fen)
            print(f"FEN: {fen}")
            print(f"True: {true_label} | Predicted: {pred_label}")
            print("-" * 60)

        if len(misclassified_idx) > 50:
            print(f"... ({len(misclassified_idx) - 50} more — see misclassified_full.csv)")

        with open(f"{run_dir}/misclassified_fens.txt", "w") as f:
            for fen in fens:
                f.write(fen + "\n")

    summary = {
        "total_positions": len(features_df),
        "misclassified": len(misclassified_idx),
        "accuracy": float(accuracy),
        "model": str(model_path),
    }
    save_json(summary, f"{run_dir}/summary.json")

    print(f"\nSaved validation run to: {run_dir}")


if __name__ == "__main__":
    main()
