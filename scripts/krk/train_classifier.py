"""
KRK WDL classifier.

Trains a decision tree to predict Win/Draw/Loss for KRK positions.
Mirrors the KPK pipeline so results are directly comparable.

Usage:
    python scripts/krk/train_classifier.py
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from endgame.utils.logger import make_run_dir, save_json, get_git_hash


DATA_PATH = "data/processed/krk/features.csv"


def main():
    print("Loading KRK features...")
    df = pd.read_csv(DATA_PATH)

    if "wdl" not in df.columns:
        raise ValueError(
            "No 'wdl' column found. Re-run scripts/krk/build_dataset.py "
            "and scripts/krk/build_features.py to regenerate with WDL."
        )

    drop_cols = [c for c in ("wdl", "dtz") if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["wdl"]

    print(f"Positions : {len(df):,}")
    print(f"Features  : {len(X.columns)}")
    print(f"Class dist:\n{y.value_counts().sort_index()}\n")

    run_dir = make_run_dir(base="logs/krk_classifier")

    meta = {
        "git_hash": get_git_hash(),
        "dataset": "krk",
        "n_samples": len(df),
        "features": list(X.columns),
        "model": "DecisionTreeClassifier",
        "experiment": "wdl_baseline",
    }
    save_json(meta, f"{run_dir}/meta.json")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=50,
        random_state=42,
    )

    print("Training...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds)

    print("=== Test Set ===")
    print(report)

    full_preds = model.predict(X)
    full_acc = (full_preds == y).mean()
    print(f"Full-dataset accuracy: {full_acc:.6f}")

    params = model.get_params()
    save_json(params, f"{run_dir}/params.json")

    metrics = classification_report(y_test, preds, output_dict=True)
    metrics["full_dataset_accuracy"] = float(full_acc)
    metrics["depth"] = model.get_depth()
    metrics["leaves"] = model.get_n_leaves()
    save_json(metrics, f"{run_dir}/metrics.json")

    summary = {
        "depth": model.get_depth(),
        "leaves": model.get_n_leaves(),
        "n_features_used": int((model.feature_importances_ > 0).sum()),
        "test_accuracy": float((preds == y_test).mean()),
        "full_dataset_accuracy": float(full_acc),
    }
    save_json(summary, f"{run_dir}/summary.json")

    tree_text = export_text(model, feature_names=list(X.columns), decimals=2)
    with open(f"{run_dir}/tree.txt", "w") as f:
        f.write(tree_text)

    import joblib
    joblib.dump(model, f"{run_dir}/model.joblib")

    print(f"\nDepth  : {model.get_depth()}")
    print(f"Leaves : {model.get_n_leaves()}")
    print(f"Saved  : {run_dir}")


if __name__ == "__main__":
    main()
