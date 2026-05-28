"""
Enumerate all legal KBNK positions and label them via Syzygy.

Iterates over all placements of WK, WB, WN, BK on distinct squares,
filters for legality, probes WDL, and writes to CSV.

Usage:
    python scripts/kbnk/build_dataset.py
"""

import os
import csv
import chess
import chess.syzygy
import multiprocessing as mp
from functools import partial


SYZYGY_PATH = "storage/syzygy/3_4_5"
OUTPUT_PATH = "data/raw/kbnk/full.csv"


def process_wk(wk, syzygy_path):
    """Process all positions for a given white king square."""
    tablebase = chess.syzygy.open_tablebase(syzygy_path)
    squares = list(chess.SQUARES)
    rows = []

    for wb in squares:
        if wb == wk:
            continue
        for wn in squares:
            if wn == wk or wn == wb:
                continue
            for bk in squares:
                if bk == wk or bk == wb or bk == wn:
                    continue
                if chess.square_distance(wk, bk) <= 1:
                    continue

                board = chess.Board(None)
                board.set_piece_at(wk, chess.Piece(chess.KING,   chess.WHITE))
                board.set_piece_at(wb, chess.Piece(chess.BISHOP, chess.WHITE))
                board.set_piece_at(wn, chess.Piece(chess.KNIGHT, chess.WHITE))
                board.set_piece_at(bk, chess.Piece(chess.KING,   chess.BLACK))

                for turn in [chess.WHITE, chess.BLACK]:
                    board.turn = turn

                    if not board.is_valid():
                        continue

                    try:
                        wdl = tablebase.probe_wdl(board)
                    except Exception:
                        continue

                    if wdl is None:
                        continue

                    wdl_norm = 2 if wdl > 0 else (-2 if wdl < 0 else 0)
                    rows.append([board.fen(), int(turn == chess.WHITE), wdl_norm])

    tablebase.close()
    return rows


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    squares = list(chess.SQUARES)
    n_cores = max(1, mp.cpu_count() - 1)
    print(f"Using {n_cores} cores")

    worker = partial(process_wk, syzygy_path=SYZYGY_PATH)

    all_rows = []
    with mp.Pool(processes=n_cores) as pool:
        results = pool.imap_unordered(worker, squares, chunksize=1)
        for i, chunk in enumerate(results):
            all_rows.extend(chunk)
            print(f"  WK {i+1}/64 done — {len(all_rows):,} rows so far")

    print(f"\nTotal legal positions: {len(all_rows):,}")
    print("Saving...")

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "turn", "wdl"])
        writer.writerows(all_rows)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
