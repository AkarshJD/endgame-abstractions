import os
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from endgame.utils.logger import make_run_dir, save_json, get_git_hash


DATA_PATH = "data/processed/kpk/features.csv"


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "dtz" not in df.columns:
        raise ValueError("DTZ column not found in KPK features file.")

    # --- Filter to winning positions only ---
    df = df[df["wdl"] == 2].copy()

    print("Filtered to winning positions only.")
    print("Remaining samples:", len(df))

    X = df.drop(columns=["wdl", "dtz"])
    y = df["dtz"]

    run_dir = make_run_dir(base="logs/kpk")

    meta = {
        "git_hash": get_git_hash(),
        "dataset": "kpk",
        "n_samples": len(df),
        "features": list(X.columns),
        "model": "DecisionTreeRegressor",
        "experiment": "dtz_regression_baseline"
    }

    save_json(meta, f"{run_dir}/meta.json")

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training model...")

    model = DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=50,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Evaluating...")

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("MAE:", mae)
    print("R2:", r2)
    print("Tree depth:", model.get_depth())
    print("Leaves:", model.get_n_leaves())

    metrics = {
        "mae": float(mae),
        "r2": float(r2),
        "depth": model.get_depth(),
        "leaves": model.get_n_leaves()
    }

    save_json(metrics, f"{run_dir}/metrics_dtz_regression.json")

    print("Saved run to:", run_dir)


if __name__ == "__main__":
    main()
