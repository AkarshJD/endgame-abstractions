import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

from endgame.utils.logger import make_run_dir, save_json, get_git_hash

DATA_PATH = "data/processed/kpk/features.csv"


def _convert_numpy(obj):
    """
    Convert numpy arrays to Python lists so they can be JSON serialized.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():

    # ----------------------------
    # Load dataset
    # ----------------------------
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["wdl"])
    y = df["wdl"]

    # ----------------------------
    # Create run directory
    # ----------------------------
    run_dir = make_run_dir()

    meta = {
        "git_hash": get_git_hash(),
        "dataset": "kpk",
        "n_samples": len(X),
        "features": list(X.columns),
        "model": "DecisionTreeClassifier",
        "experiment": "grid_search"
    }

    save_json(meta, f"{run_dir}/meta.json")

    # ----------------------------
    # Train / Test split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ----------------------------
    # Define model + grid
    # ----------------------------
    print("Building model...")

    tree = DecisionTreeClassifier(random_state=42)

    param_grid = {
        "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
        "min_samples_leaf": [1, 5, 20, 50, 100, 200],
        "min_samples_split": [2, 10, 50, 100],
        "max_features": [None, "sqrt", "log2"]
    }

    grid = GridSearchCV(
        tree,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )

    # ----------------------------
    # Run grid search
    # ----------------------------
    print("Running grid search...")
    grid.fit(X_train, y_train)

    print("\n=== Best Parameters ===")
    print(grid.best_params_)

    print("\n=== Best CV Score ===")
    print(grid.best_score_)

    # ----------------------------
    # Save full CV results
    # ----------------------------
    cv_results_clean = {
        k: _convert_numpy(v)
        for k, v in grid.cv_results_.items()
    }

    save_json(cv_results_clean, f"{run_dir}/cv_results.json")
    save_json(grid.best_params_, f"{run_dir}/best_params.json")

    # ----------------------------
    # Evaluate best model
    # ----------------------------
    best_model = grid.best_estimator_
    
    import joblib
    joblib.dump(best_model, f"{run_dir}/model.joblib")

    print("\nEvaluating on test set...")
    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred))

    metrics = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    save_json(metrics, f"{run_dir}/metrics.json")

    # ----------------------------
    # Save summary stats
    # ----------------------------
    summary = {
        "depth": best_model.get_depth(),
        "leaves": best_model.get_n_leaves(),
        "train_accuracy": best_model.score(X_train, y_train),
        "test_accuracy": best_model.score(X_test, y_test),
        "cv_best_score": grid.best_score_
    }

    save_json(summary, f"{run_dir}/summary.json")

    # ----------------------------
    # Save model params
    # ----------------------------
    save_json(best_model.get_params(), f"{run_dir}/params.json")

    # ----------------------------
    # Export tree text
    # ----------------------------
    tree_text = export_text(
        best_model,
        feature_names=list(X.columns),
        decimals=2
    )

    with open(f"{run_dir}/tree.txt", "w") as f:
        f.write(tree_text)

    print("\nTree size:")
    print("Depth:", best_model.get_depth())
    print("Leaves:", best_model.get_n_leaves())

    print("\nSaved run to:", run_dir)


if __name__ == "__main__":
    main()
