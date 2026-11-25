"""
Are KBNK draw-boundary errors learnable by a stronger model class?

Method: 5-fold matched-pair CV comparing DecisionTreeClassifier vs
RandomForestClassifier on identical folds. Key metric: DRAW recall (fidelity).

If RF DRAW recall >> DT DRAW recall: errors were model-class artifacts.
If RF DRAW recall ≈ DT DRAW recall: boundary is genuinely hard (intrinsic).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

FEATURES_CSV = "data/processed/kbnk/features.csv"
LABEL_COL = "wdl"
SKIP_COLS = {"fen", "wdl"}
N_SPLITS = 5
CLASSES = [-2, 0, 2]  # LOSS, DRAW, WIN


def per_class_recall(y_true, y_pred):
    r = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    return {"LOSS": r[0], "DRAW": r[1], "WIN": r[2]}


def main():
    print("Loading features...")
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in SKIP_COLS]
    X = df[feature_cols].values
    y = df[LABEL_COL].values
    print(f"  {len(X):,} positions, {len(feature_cols)} features")
    print(f"  Class distribution: WIN={( y==2).sum():,}  DRAW={(y==0).sum():,}  LOSS={(y==-2).sum():,}\n")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    dt_folds = []
    rf_folds = []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        dt = DecisionTreeClassifier(min_samples_leaf=50, random_state=42)
        dt.fit(X_tr, y_tr)
        dt_rec = per_class_recall(y_te, dt.predict(X_te))
        dt_folds.append(dt_rec)

        rf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=50,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        rf_rec = per_class_recall(y_te, rf.predict(X_te))
        rf_folds.append(rf_rec)

        print(f"Fold {fold + 1}/{N_SPLITS}")
        print(f"  DT   WIN={dt_rec['WIN']:.4f}  DRAW={dt_rec['DRAW']:.4f}  LOSS={dt_rec['LOSS']:.4f}")
        print(f"  RF   WIN={rf_rec['WIN']:.4f}  DRAW={rf_rec['DRAW']:.4f}  LOSS={rf_rec['LOSS']:.4f}")

    print("\n=== SUMMARY (mean ± std over 5 folds) ===")
    for label, folds in [("DecisionTree", dt_folds), ("RandomForest", rf_folds)]:
        wins  = np.array([f["WIN"]  for f in folds])
        draws = np.array([f["DRAW"] for f in folds])
        losses= np.array([f["LOSS"] for f in folds])
        print(f"\n{label}:")
        print(f"  WIN  {wins.mean():.4f} ± {wins.std():.4f}")
        print(f"  DRAW {draws.mean():.4f} ± {draws.std():.4f}")
        print(f"  LOSS {losses.mean():.4f} ± {losses.std():.4f}")

    dt_draw = np.array([f["DRAW"] for f in dt_folds]).mean()
    rf_draw = np.array([f["DRAW"] for f in rf_folds]).mean()
    delta = rf_draw - dt_draw
    print(f"\n=== KEY NUMBER ===")
    print(f"DRAW recall delta (RF - DT): {delta:+.4f}")
    if abs(delta) < 0.003:
        verdict = "NEGLIGIBLE — boundary is intrinsically hard (not model-class artifact)"
    elif delta > 0.01:
        verdict = "RF WINS — errors were model-class artifacts (decision tree underfitting)"
    else:
        verdict = "MARGINAL — ambiguous, interpret carefully"
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
