"""
Enumerate all legal KBBK positions and label them via Syzygy.

KBBK (King + 2 Bishops vs King) has a structural split:
  - Same-color bishops  → theoretical draw (insufficient material)
  - Diff-color bishops  → forced win (complementary coverage)

Bishop pairs are symmetric (B1, B2) == (B2, B1), so we enumerate
only wb1 < wb2 to avoid duplicate positions.

Usage:
    python scripts/kbbk/build_dataset.py
"""

import os
import csv
import chess
import chess.syzygy
import multiprocessing as mp
from functools import partial


SYZYGY_PATH = "storage/syzygy/3_4_5"
OUTPUT_PATH  = "data/raw/kbbk/full.csv"


def process_wk(wk, syzygy_path):
    tablebase = chess.syzygy.open_tablebase(syzygy_path)
    squares = list(chess.SQUARES)
    rows = []

    for wb1 in squares:
        if wb1 == wk:
            continue
        for wb2 in squares:
            if wb2 <= wb1 or wb2 == wk:
                continue
            for bk in squares:
                if bk == wk or bk == wb1 or bk == wb2:
                    continue
                if chess.square_distance(wk, bk) <= 1:
                    continue

                board = chess.Board(None)
                board.set_piece_at(wk,  chess.Piece(chess.KING,   chess.WHITE))
                board.set_piece_at(wb1, chess.Piece(chess.BISHOP, chess.WHITE))
                board.set_piece_at(wb2, chess.Piece(chess.BISHOP, chess.WHITE))
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
        for i, chunk in enumerate(pool.imap_unordered(worker, squares, chunksize=1)):
            all_rows.extend(chunk)
            print(f"  WK {i+1}/64 done — {len(all_rows):,} rows so far")

    print(f"\nTotal legal positions: {len(all_rows):,}")

    from collections import Counter
    dist = Counter(r[2] for r in all_rows)
    print(f"WDL distribution: {dict(sorted(dist.items()))}")

    print("Saving...")
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "turn", "wdl"])
        writer.writerows(all_rows)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
