"""
KBNK WDL validation — full dataset accuracy + misclassified positions.

Usage:
    python scripts/kbnk/validate.py
"""

import json
import joblib
import pandas as pd
from pathlib import Path

from endgame.utils.logger import make_run_dir, save_json


FEATURES_PATH = "data/processed/kbnk/features.csv"
MODEL_BASE = "logs/kbnk_classifier"


def find_latest_model(base=MODEL_BASE):
    runs = sorted(Path(base).glob("*/model.joblib"))
    if not runs:
        raise FileNotFoundError(f"No model.joblib under {base}")
    return runs[-1]


def main():
    model_path = find_latest_model()
    print(f"Model: {model_path}")

    model = joblib.load(model_path)

    with open(Path(model_path).parent / "meta.json") as f:
        feature_cols = json.load(f)["features"]

    print("Loading features...")
    features_df = pd.read_csv(FEATURES_PATH)

    has_fen = "fen" in features_df.columns
    X = features_df.reindex(columns=feature_cols)
    y_true = features_df["wdl"]

    print(f"Predicting {len(X):,} positions...")
    y_pred = model.predict(X)

    correct = y_pred == y_true.values
    accuracy = correct.mean()
    misclassified_idx = features_df[~correct].index

    print(f"\nFull-dataset accuracy : {accuracy:.6f}")
    print(f"Total positions       : {len(features_df):,}")
    print(f"Misclassified         : {len(misclassified_idx):,}")

    print("\n=== Class breakdown ===")
    labels = {2: "WIN", 0: "DRAW", -2: "LOSS"}
    for wdl_val in sorted(y_true.unique()):
        mask = y_true == wdl_val
        acc = (y_pred[mask] == wdl_val).mean()
        print(f"  {labels.get(wdl_val, str(wdl_val)):5s}: {acc:.6f}  ({mask.sum():,} positions)")

    run_dir = make_run_dir(base="logs/kbnk_validation")

    features_df["prediction"] = y_pred
    features_df["correct"] = correct
    features_df.to_csv(f"{run_dir}/full_predictions.csv", index=False)

    if has_fen:
        misclassified_df = features_df.loc[misclassified_idx, ["fen", "wdl", "prediction"]].copy()
        misclassified_df["true_label"]  = misclassified_df["wdl"].map(labels)
        misclassified_df["pred_label"]  = misclassified_df["prediction"].map(labels)
        misclassified_df.to_csv(f"{run_dir}/misclassified_full.csv", index=False)

        print("\n===== MISCLASSIFIED FEN POSITIONS (first 30) =====\n")
        for _, row in misclassified_df.head(30).iterrows():
            print(f"FEN: {row['fen']}")
            print(f"True: {row['true_label']} | Predicted: {row['pred_label']}")
            print("-" * 60)

        if len(misclassified_idx) > 30:
            print(f"... ({len(misclassified_idx) - 30} more in misclassified_full.csv)")

        # Confusion breakdown: which transitions are most common
        print("\n=== Misclassification transitions ===")
        transitions = misclassified_df.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False)
        print(transitions.to_string())

    summary = {
        "total_positions": len(features_df),
        "misclassified": len(misclassified_idx),
        "accuracy": float(accuracy),
        "model": str(model_path),
    }
    save_json(summary, f"{run_dir}/summary.json")
    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
