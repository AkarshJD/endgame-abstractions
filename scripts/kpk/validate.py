import pandas as pd
import joblib

from endgame.utils.logger import make_run_dir, save_json


MODEL_PATH = "logs/kpk/2026-02-10_21-55-59/model.joblib"
FEATURE_PATH = "data/processed/kpk/features.csv"
FULL_PATH = "data/processed/kpk/full.csv"


def main():

    print("Loading features...")
    features_df = pd.read_csv(FEATURE_PATH)

    print("Loading full dataset (with FEN)...")
    full_df = pd.read_csv(FULL_PATH)

    X = features_df.drop(columns=["wdl"])
    y_true = features_df["wdl"]

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

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

    print("\n===== MISCLASSIFIED FEN POSITIONS =====\n")

    fens = []

    for idx in misclassified_idx:
        fen = full_df.loc[idx, "fen"]
        true_label = full_df.loc[idx, "wdl"]
        pred_label = features_df.loc[idx, "prediction"]

        fens.append(fen)

        print(f"FEN: {fen}")
        print(f"True: {true_label} | Predicted: {pred_label}")
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
