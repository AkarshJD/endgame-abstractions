import os
import pandas as pd
import chess

from endgame.features.krk_geom import compute_krk_features


RAW_PATH = "data/raw/krk/full.csv"
OUTPUT_PATH = "data/processed/krk/features.csv"


def main():

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading raw dataset...")
    df = pd.read_csv(RAW_PATH)

    rows = []

    print("Extracting features...")

    for _, row in df.iterrows():

        board = chess.Board(row["fen"])

        features = compute_krk_features(board)

        features["dtz"] = row["dtz"]

        rows.append(features)

    feature_df = pd.DataFrame(rows)

    print("Saving processed features...")

    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("Done.")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
