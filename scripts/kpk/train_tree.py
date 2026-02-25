import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from endgame.utils.logger import make_run_dir, save_json, get_git_hash


def main():

    # ----------------------------
    # Load dataset
    # ----------------------------
    path = "data/processed/kpk/features.csv"
    df = pd.read_csv(path)

    X = df.drop(columns=["wdl", "dtz"])
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
        "experiment": "baseline_static"
    }

    save_json(meta, f"{run_dir}/meta.json")

    # ----------------------------
    # Train / Test split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ----------------------------
    # Train model
    # ----------------------------
    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ----------------------------
    # Evaluate
    # ----------------------------
    preds = model.predict(X_test)

    report = classification_report(y_test, preds)

    print("=== Evaluation ===")
    print(report)

    # ----------------------------
    # Save params
    # ----------------------------
    params = model.get_params()
    save_json(params, f"{run_dir}/params.json")

    # ----------------------------
    # Save metrics
    # ----------------------------
    metrics = classification_report(
        y_test,
        preds,
        output_dict=True
    )

    save_json(metrics, f"{run_dir}/metrics.json")

    # ----------------------------
    # Export tree
    # ----------------------------
    tree_text = export_text(
        model,
        feature_names=list(X.columns),
        decimals=2
    )

    with open(f"{run_dir}/tree.txt", "w") as f:
        f.write(tree_text)

    print("Saved run to:", run_dir)


if __name__ == "__main__":
    main()
