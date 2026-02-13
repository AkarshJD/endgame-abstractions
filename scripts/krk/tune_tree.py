import os
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from endgame.utils.logger import make_run_dir, save_json, get_git_hash


DATA_PATH = "data/processed/krk/features.csv"


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["dtz"])
    y = df["dtz"]

    run_dir = make_run_dir(base="logs/krk")

    meta = {
        "git_hash": get_git_hash(),
        "dataset": "krk",
        "n_samples": len(df),
        "features": list(X.columns),
        "model": "DecisionTreeRegressor",
        "experiment": "grid_tuning"
    }

    save_json(meta, f"{run_dir}/meta.json")

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    tree = DecisionTreeRegressor(random_state=42)

    param_grid = {
        "max_depth": [8, 10, 12, 15, 20, None],
        "min_samples_leaf": [5, 10, 20, 50, 100]
    }

    print("Running grid search...")
    grid = GridSearchCV(
        tree,
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\nTest Metrics:")
    print("MAE:", mae)
    print("R2:", r2)
    print("Depth:", best_model.get_depth())
    print("Leaves:", best_model.get_n_leaves())

    metrics = {
        "mae": float(mae),
        "r2": float(r2),
        "depth": best_model.get_depth(),
        "leaves": best_model.get_n_leaves(),
        "best_params": grid.best_params_
    }

    save_json(metrics, f"{run_dir}/metrics.json")

    print("Saved run to:", run_dir)


if __name__ == "__main__":
    main()
