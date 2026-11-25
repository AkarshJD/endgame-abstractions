"""
KPK WDL classifier.

Same setup as the other endgames' train_classifier.py (random 20% test split,
min_samples_leaf=50, no depth limit), so the results table compares like with
like. It also reports a pawn-file holdout: train on pawns on files c-h, test
on files a and b. Files a and b mirror h and g, which stay in training, so the
holdout mostly shows how left-right symmetric the features are.

Usage:
    python scripts/kpk/train_classifier.py
"""

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from endgame.utils.logger import get_git_hash, make_run_dir, save_json

DATA_PATH = "data/processed/kpk/features.csv"
HOLDOUT_FILES = (0, 1)  # a- and b-file pawns


def new_tree():
    return DecisionTreeClassifier(max_depth=None, min_samples_leaf=50, random_state=42)


def main():
    print("Loading KPK features...")
    df = pd.read_csv(DATA_PATH)
    if df.isna().any().any():
        raise ValueError(f"{DATA_PATH} has missing values; rebuild it with scripts/kpk/build_features.py")

    drop_cols = [c for c in ("wdl", "dtz") if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["wdl"]

    print(f"Positions : {len(df):,}")
    print(f"Features  : {len(X.columns)}")
    print(f"Class dist:\n{y.value_counts().sort_index()}\n")

    run_dir = make_run_dir(base="logs/kpk_classifier")
    save_json(
        {
            "git_hash": get_git_hash(),
            "dataset": "kpk",
            "n_samples": len(df),
            "features": list(X.columns),
            "model": "DecisionTreeClassifier",
            "experiment": "wdl_baseline",
        },
        f"{run_dir}/meta.json",
    )

    # ---- random split, as for the other endgames ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = new_tree()
    print("Training...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("=== Test Set ===")
    print(classification_report(y_test, preds))
    full_acc = float((model.predict(X) == y).mean())
    print(f"Full-dataset accuracy: {full_acc:.6f}")

    # ---- pawn-file holdout ----
    held = df["pawn_file"].isin(HOLDOUT_FILES)
    holdout_model = new_tree().fit(X[~held], y[~held])
    holdout_acc = float(holdout_model.score(X[held], y[held]))
    print(f"Pawn-file holdout (train c-h, test a-b): {holdout_acc:.6f} on {int(held.sum()):,} positions")

    save_json(model.get_params(), f"{run_dir}/params.json")
    metrics = classification_report(y_test, preds, output_dict=True)
    metrics["full_dataset_accuracy"] = full_acc
    metrics["depth"] = model.get_depth()
    metrics["leaves"] = model.get_n_leaves()
    save_json(metrics, f"{run_dir}/metrics.json")

    save_json(
        {
            "depth": model.get_depth(),
            "leaves": model.get_n_leaves(),
            "n_features_used": int((model.feature_importances_ > 0).sum()),
            "test_accuracy": float((preds == y_test).mean()),
            "full_dataset_accuracy": full_acc,
            "pawn_file_holdout": {
                "test_files": list(HOLDOUT_FILES),
                "n_test": int(held.sum()),
                "test_accuracy": holdout_acc,
                "depth": holdout_model.get_depth(),
                "leaves": holdout_model.get_n_leaves(),
            },
        },
        f"{run_dir}/summary.json",
    )

    with open(f"{run_dir}/tree.txt", "w") as f:
        f.write(export_text(model, feature_names=list(X.columns), decimals=2))

    import joblib

    joblib.dump(model, f"{run_dir}/model.joblib")

    print(f"\nDepth  : {model.get_depth()}")
    print(f"Leaves : {model.get_n_leaves()}")
    print(f"Saved  : {run_dir}")


if __name__ == "__main__":
    main()
