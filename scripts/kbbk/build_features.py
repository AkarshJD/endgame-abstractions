"""
Extract geometric features for KBBK positions.

Usage:
    python scripts/kbbk/build_features.py
"""

import os
import math
import multiprocessing as mp
import pandas as pd
import chess

from endgame.features.kbbk_geom import compute_kbbk_features


RAW_PATH    = "data/raw/kbbk/full.csv"
OUTPUT_PATH = "data/processed/kbbk/features.csv"
SAMPLE_SIZE = 2_000_000
RANDOM_SEED = 42
N_WORKERS   = max(1, mp.cpu_count() - 1)


def process_chunk(rows):
    out = []
    errors = 0
    for fen, wdl in rows:
        try:
            board = chess.Board(fen)
            features = compute_kbbk_features(board)
            features["fen"] = fen
            features["wdl"] = wdl
            out.append(features)
        except Exception:
            errors += 1
    return out, errors


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading raw dataset...")
    df = pd.read_csv(RAW_PATH, usecols=["fen", "wdl"])
    total = len(df)
    print(f"Total positions: {total:,}")
    print(f"WDL dist (full):\n{df['wdl'].value_counts().sort_index()}\n")

    n = min(SAMPLE_SIZE, total)
    if total > n:
        df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"Sampled {n:,} positions")
    print(f"WDL dist (sample):\n{df['wdl'].value_counts().sort_index()}\n")

    pairs = list(zip(df["fen"], df["wdl"]))
    chunk_size = math.ceil(len(pairs) / N_WORKERS)
    chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    print(f"Extracting features with {N_WORKERS} workers ({len(chunks)} chunks)...")
    all_rows = []
    total_errors = 0

    with mp.Pool(processes=N_WORKERS) as pool:
        for i, (chunk_rows, errors) in enumerate(pool.imap_unordered(process_chunk, chunks)):
            all_rows.extend(chunk_rows)
            total_errors += errors
            print(f"  chunk {i+1}/{len(chunks)} done — {len(all_rows):,} features so far")

    print(f"\nDone. Errors: {total_errors}")

    feature_df = pd.DataFrame(all_rows)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Shape  : {feature_df.shape}")
    print(f"WDL dist:\n{feature_df['wdl'].value_counts().sort_index()}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
