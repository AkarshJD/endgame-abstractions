"""
Extract geometric features for KBNK positions.

Subsamples 2M positions from the full dataset (24.5M) to keep
feature matrix in memory (~1.4 GB vs ~7 GB for the full set).
Uses multiprocessing to parallelize FEN parsing + feature extraction.

Usage:
    python scripts/kbnk/build_features.py
"""

import os
import math
import multiprocessing as mp
import pandas as pd
import chess

from endgame.features.kbnk_geom import compute_kbnk_features


RAW_PATH      = "data/raw/kbnk/full.csv"
OUTPUT_PATH   = "data/processed/kbnk/features.csv"
SAMPLE_SIZE   = 2_000_000
RANDOM_SEED   = 42
N_WORKERS     = max(1, mp.cpu_count() - 1)


def process_chunk(rows):
    """Feature-extract a list of (fen, wdl) tuples. Returns list of feature dicts."""
    out = []
    errors = 0
    for fen, wdl in rows:
        try:
            board = chess.Board(fen)
            features = compute_kbnk_features(board)
            features["fen"] = fen
            features["wdl"] = wdl
            out.append(features)
        except Exception:
            errors += 1
    return out, errors


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Loading raw dataset (sampling {SAMPLE_SIZE:,} of ~24.5M)...")
    df = pd.read_csv(RAW_PATH, usecols=["fen", "wdl"])
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Sample WDL dist:\n{df['wdl'].value_counts().sort_index()}\n")

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
