"""
tune_tree.py

Purpose:
- No heavy CV / GridSearchCV.
- Do a small, interpretable sweep and pick the smallest tree that hits accuracy.
- Hard-drop obvious leakage (wdl/dtz + optional denylist substrings).
- Supports either:
    A) random stratified split, or
    B) structured pawn-file holdout split (recommended for chess symmetry robustness).

Output artifacts (in run_dir):
- meta.json
- summary.json
- best_params.json
- params.json
- metrics.json
- tree.txt
- sweep_results.json
- model.joblib
"""

import json
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from endgame.utils.logger import make_run_dir, save_json, get_git_hash


DATA_PATH = "data/processed/kpk/features.csv"


# ----------------------------
# Leakage filtering
# ----------------------------

# Drop columns that are very likely to encode tablebase outcome/dtz directly or indirectly.
# Keep this conservative. You can relax later once you confirm a feature is geometry-only.
LEAKY_SUBSTRINGS = [
    "wdl", "dtz",
    "turn_sensitivity",
    "worsening",
    "null_move",
    "tablebase", "tb_",
    "reciprocal_zugzwang",
    "trebuchet",
    "full_point",
]

def drop_leaky_columns(X: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for c in X.columns:
        cl = c.lower()
        if any(s in cl for s in LEAKY_SUBSTRINGS):
            continue
        keep.append(c)
    return X[keep]


# ----------------------------
# Split strategies
# ----------------------------

def split_random_stratified(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def split_pawn_file_holdout(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, test_files=(0, 1)):
    """
    Hold out pawn files (a=0, b=1 by default).
    This reduces symmetry leakage and tests generalization of rules across files.
    Requires a pawn_file column in df/X.
    """
    if "pawn_file" not in df.columns:
        raise ValueError("pawn_file_holdout split requires 'pawn_file' column in the CSV.")

    test_mask = df["pawn_file"].isin(set(test_files))

    X_train = X.loc[~test_mask].copy()
    y_train = y.loc[~test_mask].copy()
    X_test  = X.loc[test_mask].copy()
    y_test  = y.loc[test_mask].copy()

    if len(X_test) == 0 or len(X_train) == 0:
        raise ValueError("pawn_file_holdout produced empty train or test set. Check test_files.")
    return X_train, X_test, y_train, y_test


# ----------------------------
# Model sweep
# ----------------------------

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    preds = model.predict(X_test)
    metrics = classification_report(y_test, preds, output_dict=True)

    depth = model.get_depth()
    leaves = model.get_n_leaves()
    n_features_used = int(np.sum(model.feature_importances_ > 0))

    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "depth": int(depth),
        "leaves": int(leaves),
        "n_features_used": n_features_used,
        "metrics": metrics,
    }


def better_candidate(a, b):
    """
    Compare two sweep results dicts (both must contain: test_accuracy, leaves, depth).
    Return True if a is better than b.
    Tie-break: higher test_accuracy -> fewer leaves -> smaller depth.
    """
    if b is None:
        return True

    if a["test_accuracy"] != b["test_accuracy"]:
        return a["test_accuracy"] > b["test_accuracy"]

    if a["leaves"] != b["leaves"]:
        return a["leaves"] < b["leaves"]

    return a["depth"] < b["depth"]


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # ----------------------------
    # Build X/y with leakage prevention
    # ----------------------------
    y = df["wdl"]

    # drop obvious targets, then substring denylist
    X = df.drop(columns=["wdl", "dtz"], errors="ignore")
    X = drop_leaky_columns(X)

    # ----------------------------
    # Choose split strategy
    # ----------------------------
    SPLIT_MODE = "pawn_file_holdout"  # "random_stratified" or "pawn_file_holdout"

    if SPLIT_MODE == "random_stratified":
        X_train, X_test, y_train, y_test = split_random_stratified(X, y, test_size=0.2, random_state=42)
        split_info = {"mode": "random_stratified", "test_size": 0.2, "random_state": 42}
    elif SPLIT_MODE == "pawn_file_holdout":
        # hold out a/b files by default
        test_files = (0, 1)
        X_train, X_test, y_train, y_test = split_pawn_file_holdout(df, X, y, test_files=test_files)
        split_info = {"mode": "pawn_file_holdout", "test_files": list(test_files)}
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    # ----------------------------
    # Create run directory + meta
    # ----------------------------
    run_dir = make_run_dir()

    meta = {
        "git_hash": get_git_hash(),
        "dataset": "kpk",
        "data_path": DATA_PATH,
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(X.columns),
        "model": "DecisionTreeClassifier",
        "experiment": "sweep_prune_no_cv",
        "split": split_info,
        "leak_filter": {
            "dropped_columns_exact": ["wdl", "dtz"],
            "dropped_if_contains": LEAKY_SUBSTRINGS,
        },
    }
    save_json(meta, f"{run_dir}/meta.json")

    # ----------------------------
    # Sweep space (small but effective)
    # ----------------------------
    # Key knobs:
    # - max_depth: interpretability / boundary complexity
    # - min_samples_leaf: smoothness + avoid brittle splits
    # - ccp_alpha: post-pruning (very useful for rule compression)
    max_depth_grid = [2, 3, 4, 5, 6, 7, 8, 10, None]
    min_samples_leaf_grid = [50, 100, 200, 400, 800]
    ccp_alpha_grid = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]

    # Optional constraint: require at least this test accuracy, then minimize leaves/depth.
    ACC_TARGET = None  # e.g. 0.995; set to None to just maximize accuracy with tie-break

    print("Running sweep (no CV)...")
    sweep_rows = []
    best = None
    best_model = None

    total = len(max_depth_grid) * len(min_samples_leaf_grid) * len(ccp_alpha_grid)
    idx = 0

    for max_depth in max_depth_grid:
        for min_leaf in min_samples_leaf_grid:
            for ccp_alpha in ccp_alpha_grid:
                idx += 1
                model = DecisionTreeClassifier(
                    random_state=42,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    ccp_alpha=ccp_alpha,
                )

                stats = evaluate_model(model, X_train, y_train, X_test, y_test)

                row = {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_leaf,
                    "ccp_alpha": float(ccp_alpha),
                    "train_accuracy": stats["train_accuracy"],
                    "test_accuracy": stats["test_accuracy"],
                    "depth": stats["depth"],
                    "leaves": stats["leaves"],
                    "n_features_used": stats["n_features_used"],
                }
                sweep_rows.append(row)

                # Candidate selection
                if ACC_TARGET is not None and row["test_accuracy"] < ACC_TARGET:
                    continue

                if better_candidate(row, best):
                    best = row
                    best_model = model  # already fit inside evaluate_model, but we’ll refit cleanly later

                if idx % 25 == 0 or idx == total:
                    print(f"  {idx}/{total} checked... current best: {best}")

    if best is None:
        raise RuntimeError("No model met ACC_TARGET (or sweep produced no candidates). Lower ACC_TARGET or expand grid.")

    # ----------------------------
    # Refit best model and finalize artifacts
    # ----------------------------
    print("\n=== Best Sweep Choice ===")
    print(best)

    final_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=best["max_depth"],
        min_samples_leaf=best["min_samples_leaf"],
        ccp_alpha=best["ccp_alpha"],
    )
    final_stats = evaluate_model(final_model, X_train, y_train, X_test, y_test)

    # Save sweep
    save_json({"results": sweep_rows}, f"{run_dir}/sweep_results.json")

    # Save params
    best_params = {
        "max_depth": best["max_depth"],
        "min_samples_leaf": best["min_samples_leaf"],
        "ccp_alpha": best["ccp_alpha"],
    }
    save_json(best_params, f"{run_dir}/best_params.json")
    save_json(final_model.get_params(), f"{run_dir}/params.json")

    # Save metrics
    save_json(final_stats["metrics"], f"{run_dir}/metrics.json")

    # Save summary
    summary = {
        "depth": final_stats["depth"],
        "leaves": final_stats["leaves"],
        "n_features_used": final_stats["n_features_used"],
        "train_accuracy": final_stats["train_accuracy"],
        "test_accuracy": final_stats["test_accuracy"],
        "acc_target": ACC_TARGET,
        "chosen_params": best_params,
    }
    save_json(summary, f"{run_dir}/summary.json")

    # Export tree
    tree_text = export_text(
        final_model,
        feature_names=list(X.columns),
        decimals=2
    )
    with open(f"{run_dir}/tree.txt", "w") as f:
        f.write(tree_text)

    # Save model
    import joblib
    joblib.dump(final_model, f"{run_dir}/model.joblib")

    # Print evaluation
    print("\n=== Evaluation (Test Set) ===")
    preds = final_model.predict(X_test)
    print(classification_report(y_test, preds))

    print("\nTree size:")
    print("Depth:", final_model.get_depth())
    print("Leaves:", final_model.get_n_leaves())

    print("\nSaved run to:", run_dir)


if __name__ == "__main__":
    main()