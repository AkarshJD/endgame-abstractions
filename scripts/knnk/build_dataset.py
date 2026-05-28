"""
Enumerate all legal KNNK positions and label them via Syzygy.

KNNK (King + 2 Knights vs King) is the canonical "theoretical draw" endgame:
white cannot force checkmate with best play, but zugzwang positions exist
when it is black to move. WDL is therefore almost entirely 0 (draw) with
a small minority of -2 positions (black to move, forced loss).

Knight pairs are symmetric (N1, N2) == (N2, N1), so we enumerate only
wn1 < wn2 to avoid duplicate positions.

Usage:
    python scripts/knnk/build_dataset.py
"""

import os
import csv
import chess
import chess.syzygy
import multiprocessing as mp
from functools import partial


SYZYGY_PATH = "storage/syzygy/3_4_5"
OUTPUT_PATH  = "data/raw/knnk/full.csv"


def process_wk(wk, syzygy_path):
    """Process all KNNK positions for a given white king square."""
    tablebase = chess.syzygy.open_tablebase(syzygy_path)
    squares = list(chess.SQUARES)
    rows = []

    for wn1 in squares:
        if wn1 == wk:
            continue
        for wn2 in squares:
            # Enforce wn1 < wn2 to avoid symmetric duplicates
            if wn2 <= wn1 or wn2 == wk:
                continue
            for bk in squares:
                if bk == wk or bk == wn1 or bk == wn2:
                    continue
                if chess.square_distance(wk, bk) <= 1:
                    continue

                board = chess.Board(None)
                board.set_piece_at(wk,  chess.Piece(chess.KING,   chess.WHITE))
                board.set_piece_at(wn1, chess.Piece(chess.KNIGHT, chess.WHITE))
                board.set_piece_at(wn2, chess.Piece(chess.KNIGHT, chess.WHITE))
                board.set_piece_at(bk,  chess.Piece(chess.KING,   chess.BLACK))

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

    wdl_vals = [r[2] for r in all_rows]
    from collections import Counter
    dist = Counter(wdl_vals)
    print(f"WDL distribution: {dict(sorted(dist.items()))}")

    print("Saving...")
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "turn", "wdl"])
        writer.writerows(all_rows)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
