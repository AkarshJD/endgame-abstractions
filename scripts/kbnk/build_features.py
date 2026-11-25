"""
Extract geometric features for KBNK positions.

Subsamples 2M positions from the full dataset (24.5M) to keep
feature matrix in memory (~1.4 GB vs ~7 GB for the full set).
Uses multiprocessing to parallelize FEN parsing + feature extraction.

Usage:
    # Baseline (no DTZ — same as original):
    python scripts/kbnk/build_features.py

    # Intervention (adds DTZ column — requires .rtbz files):
    python scripts/kbnk/build_features.py --with-dtz

The --with-dtz flag is the single toggle that separates the baseline from the
faithfulness-decomposition intervention. Both code paths share the same
feature extraction logic; only the DTZ probing pass differs.

Output:
    Baseline  → data/processed/kbnk/features.csv
    With DTZ  → data/processed/kbnk/features_dtz.csv
"""

import argparse
import os
import math
import multiprocessing as mp
import pandas as pd
import chess
import chess.syzygy

from endgame.features.kbnk_geom import compute_kbnk_features


RAW_PATH      = "data/raw/kbnk/full.csv"
OUTPUT_PATH   = "data/processed/kbnk/features.csv"
OUTPUT_DTZ    = "data/processed/kbnk/features_dtz.csv"
SYZYGY_PATH   = "storage/syzygy/3_4_5"
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


def probe_dtz_chunk(args):
    """Probe DTZ for a list of FEN strings. Returns list of int|None."""
    fens, syzygy_path = args
    tb = chess.syzygy.open_tablebase(syzygy_path)
    results = []
    for fen in fens:
        try:
            dtz = tb.probe_dtz(chess.Board(fen))
        except Exception:
            dtz = None
        results.append(dtz)
    tb.close()
    return results


def probe_dtz_parallel(fens, syzygy_path=SYZYGY_PATH):
    """Probe DTZ for all FENs using multiprocessing. Returns list of int|None."""
    chunk_size = math.ceil(len(fens) / N_WORKERS)
    chunks = [fens[i : i + chunk_size] for i in range(0, len(fens), chunk_size)]
    args = [(chunk, syzygy_path) for chunk in chunks]
    results = []
    print(f"  Probing DTZ with {N_WORKERS} workers ({len(chunks)} chunks)...")
    with mp.Pool(processes=N_WORKERS) as pool:
        for i, chunk_result in enumerate(pool.imap(probe_dtz_chunk, args)):
            results.extend(chunk_result)
            print(f"    DTZ chunk {i+1}/{len(chunks)} done — {len(results):,} probed")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-dtz",
        action="store_true",
        help="Also probe DTZ from Syzygy .rtbz files and include as a feature column.",
    )
    args = parser.parse_args()

    output_path = OUTPUT_DTZ if args.with_dtz else OUTPUT_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    print(f"\nFeature extraction done. Errors: {total_errors}")

    feature_df = pd.DataFrame(all_rows)

    if args.with_dtz:
        print("\nProbing DTZ (faithfulness-decomposition intervention)...")
        fens = feature_df["fen"].tolist()
        dtz_values = probe_dtz_parallel(fens)
        feature_df["dtz"] = dtz_values
        n_failed = feature_df["dtz"].isna().sum()
        if n_failed:
            print(f"  DTZ probe failed for {n_failed:,} positions — set to NaN")
        print(f"  DTZ range: [{feature_df['dtz'].min()}, {feature_df['dtz'].max()}]")

    feature_df.to_csv(output_path, index=False)

    print(f"\nShape  : {feature_df.shape}")
    print(f"WDL dist:\n{feature_df['wdl'].value_counts().sort_index()}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
