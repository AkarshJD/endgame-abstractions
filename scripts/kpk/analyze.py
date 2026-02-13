import sys
import os
import pandas as pd
import numpy as np


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("python analyze_kpk_misclassified.py path/to/misclassified.csv")
        return

    path = sys.argv[1]

    print("Loading misclassified file...")
    df = pd.read_csv(path)

    print("\n===== BASIC INFO =====")
    print("Total misclassified:", len(df))
    print("Columns:", len(df.columns))

    # ============================================================
    # CLASS DISTRIBUTION
    # ============================================================

    print("\n===== TRUE LABEL DISTRIBUTION =====")
    print(df["wdl"].value_counts())

    if "prediction" in df.columns:
        print("\n===== PREDICTED LABEL DISTRIBUTION =====")
        print(df["prediction"].value_counts())

        print("\n===== CONFUSION COUNTS =====")
        print(pd.crosstab(df["wdl"], df["prediction"]))

    # ============================================================
    # TURN DISTRIBUTION
    # ============================================================

    if "turn" in df.columns:
        print("\n===== TURN DISTRIBUTION =====")
        print(df["turn"].value_counts())

    # ============================================================
    # LOW VARIANCE FEATURES
    # ============================================================

    print("\n===== LOW VARIANCE FEATURES =====")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    variances = df[numeric_cols].var().sort_values()

    print(variances.head(15))

    # ============================================================
    # MOST COMMON FEATURE VALUES
    # ============================================================

    print("\n===== MOST COMMON FEATURE VALUES (>=60%) =====")

    for col in numeric_cols:
        top = df[col].value_counts().head(1)
        if len(top) > 0:
            value = top.index[0]
            count = top.iloc[0]
            if count >= len(df) * 0.6:
                print(f"{col}: {value} ({count}/{len(df)})")

    # ============================================================
    # CORRELATION WITH TRUE LABEL
    # ============================================================

    print("\n===== CORRELATION WITH TRUE LABEL =====")

    correlations = df[numeric_cols].corr()["wdl"].sort_values()
    print(correlations.head(10))
    print(correlations.tail(10))

    # ============================================================
    # PAWN RANK DISTRIBUTION
    # ============================================================

    if "pawn_rank" in df.columns:
        print("\n===== PAWN RANK DISTRIBUTION =====")
        print(df["pawn_rank"].value_counts())

    # ============================================================
    # DISTANCE FEATURE SUMMARY
    # ============================================================

    print("\n===== DISTANCE FEATURES SUMMARY =====")

    distance_cols = [c for c in df.columns if "d_" in c or "e2_" in c]
    if distance_cols:
        print(df[distance_cols].describe().T[["mean", "min", "max"]])

    # ============================================================
    # UNIQUE GEOMETRY CHECK (IGNORE TURN)
    # ============================================================

    print("\n===== UNIQUE GEOMETRY IGNORING TURN =====")

    geometry_cols = [
        c for c in df.columns
        if c not in ["wdl", "prediction", "correct", "turn"]
    ]

    unique_geometries = df.groupby(geometry_cols).size()
    print("Unique geometric configs (ignoring turn):", len(unique_geometries))

    # ============================================================
    # PRINT ALL MISCLASSIFIED POSITIONS
    # ============================================================

    print("\n===== ALL MISCLASSIFIED POSITIONS =====")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df)

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================

    output_dir = "analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save full raw file
    df.to_csv(f"{output_dir}/misclassified_full.csv", index=False)

    # Save geometry-only (drop prediction helpers)
    drop_cols = [c for c in ["prediction", "correct"] if c in df.columns]
    df_geometry = df.drop(columns=drop_cols)
    df_geometry.to_csv(
        f"{output_dir}/misclassified_geometry_only.csv",
        index=False
    )

    # Save sorted version for easier inspection
    sort_cols = [c for c in ["pawn_rank", "file_gap", "d_promo"] if c in df.columns]
    if sort_cols:
        df_sorted = df.sort_values(by=sort_cols)
        df_sorted.to_csv(
            f"{output_dir}/misclassified_sorted.csv",
            index=False
        )

    # Save readable text dump
    with open(f"{output_dir}/misclassified_readable.txt", "w") as f:
        f.write(df.to_string(index=False))

    print(f"\nSaved analysis outputs to: {output_dir}/")


if __name__ == "__main__":
    main()
