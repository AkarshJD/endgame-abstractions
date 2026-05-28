"""
KNNK WDL classifier.

Usage:
    python scripts/knnk/train_classifier.py
"""

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from endgame.utils.logger import make_run_dir, save_json, get_git_hash


DATA_PATH = "data/processed/knnk/features.csv"


def main():
    print("Loading KNNK features...")
    df = pd.read_csv(DATA_PATH)

    drop_cols = [c for c in ("wdl", "fen") if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["wdl"]

    print(f"Positions : {len(df):,}")
    print(f"Features  : {len(X.columns)}")
    print(f"Class dist:\n{y.value_counts().sort_index()}\n")

    run_dir = make_run_dir(base="logs/knnk_classifier")

    meta = {
        "git_hash": get_git_hash(),
        "dataset": "knnk",
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
    print("=== Test Set ===")
    print(classification_report(y_test, preds))

    full_preds = model.predict(X)
    full_acc = (full_preds == y).mean()
    print(f"Full-dataset accuracy: {full_acc:.6f}")
    print(f"Depth  : {model.get_depth()}")
    print(f"Leaves : {model.get_n_leaves()}")

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

    joblib.dump(model, f"{run_dir}/model.joblib")
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
