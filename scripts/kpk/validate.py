import json
from pathlib import Path

import joblib
import pandas as pd

from endgame.utils.logger import make_run_dir, save_json


MODEL_BASE = "logs/kpk_classifier"  # written by scripts/kpk/train_classifier.py
FEATURE_PATH = "data/processed/kpk/features.csv"
FULL_PATH = "data/processed/kpk/full.csv"
N_PRINT = 20  # misclassified FENs to print; all are saved to the run directory


def find_latest_model(base=MODEL_BASE):
    runs = sorted(Path(base).glob("*/model.joblib"))
    if not runs:
        raise FileNotFoundError(f"No model.joblib under {base}; run scripts/kpk/train_classifier.py first")
    return runs[-1]


def main():

    model_path = find_latest_model()
    print(f"Model: {model_path}")
    feature_cols = json.loads((model_path.parent / "meta.json").read_text())["features"]

    print("Loading features...")
    features_df = pd.read_csv(FEATURE_PATH)

    print("Loading full dataset (with FEN)...")
    full_df = pd.read_csv(FULL_PATH)

    X = features_df.reindex(columns=feature_cols)
    y_true = features_df["wdl"]

    print("Loading model...")
    model = joblib.load(model_path)

    print("Predicting full dataset...")
    y_pred = model.predict(X)

    features_df["prediction"] = y_pred
    features_df["correct"] = features_df["prediction"] == y_true

    misclassified_idx = features_df[~features_df["correct"]].index
    misclassified_features = features_df.loc[misclassified_idx]
    misclassified_full = full_df.loc[misclassified_idx]

    accuracy = features_df["correct"].mean()

    print("\nFull dataset accuracy:", accuracy)
    print("Total positions:", len(features_df))
    print("Misclassified:", len(misclassified_idx))

    # ============================================================
    # PRINT MISCLASSIFIED FENS
    # ============================================================

    print(f"\n===== MISCLASSIFIED FEN POSITIONS (first {N_PRINT}) =====\n")

    fens = []

    for i, idx in enumerate(misclassified_idx):
        fen = full_df.loc[idx, "fen"]
        fens.append(fen)
        if i < N_PRINT:
            print(f"FEN: {fen}")
            print(f"True: {full_df.loc[idx, 'wdl']} | Predicted: {features_df.loc[idx, 'prediction']}")
            print("-" * 60)

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    run_dir = make_run_dir(base="logs/kpk_validation")

    features_df.to_csv(f"{run_dir}/full_predictions.csv", index=False)
    misclassified_features.to_csv(f"{run_dir}/misclassified_features.csv", index=False)
    misclassified_full.to_csv(f"{run_dir}/misclassified_full.csv", index=False)

    # Save FEN list
    with open(f"{run_dir}/misclassified_fens.txt", "w") as f:
        for fen in fens:
            f.write(fen + "\n")

    summary = {
        "total_positions": len(features_df),
        "misclassified": len(misclassified_idx),
        "accuracy": float(accuracy)
    }

    save_json(summary, f"{run_dir}/summary.json")

    print("\nSaved validation run to:", run_dir)


if __name__ == "__main__":
    main()
