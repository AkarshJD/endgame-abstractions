import csv
import chess
import pandas as pd

from pathlib import Path

from endgame.features.kpk_geom import extract_kpk_features


IN_FILE = Path("data/processed/kpk/full.csv")
OUT_FILE = Path("data/processed/kpk/features.csv")


def main():

    df = pd.read_csv(IN_FILE)

    rows = []

    for _, row in df.iterrows():

        board = chess.Board(row["fen"])

        feats = extract_kpk_features(board)

        feats["wdl"] = row["wdl"]

        rows.append(feats)

    out = pd.DataFrame(rows)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(OUT_FILE, index=False)

    print("Saved:", OUT_FILE)
    print("Shape:", out.shape)


if __name__ == "__main__":
    main()
