"""
Count cursed-win (raw WDL=+1) and blessed-loss (raw WDL=-1) positions in
the KBBKN (King+2Bishops vs King+Knight) endgame.

KBBKN is the positive control: Wikipedia confirms max DTC = 66 full moves,
which exceeds the 50-move rule limit (50 full moves = 100 half-moves).
A non-zero cursed count here validates the method; zero in KBNK/KRK/KPK
confirms the mechanism is absent in those endgames.

Method: enumerate all legal KBBKN positions directly (no CSV required),
probe raw WDL via Syzygy. Run in parallel by white king square.
"""

import argparse
import chess
import chess.syzygy
import math
import multiprocessing as mp
import sys

SYZYGY = "storage/syzygy/3_4_5"
LABELS = {
    2:  "Clean win",
    1:  "Cursed win  (clock-dependent)",
    0:  "Draw",
    -1: "Blessed loss (clock-dependent)",
    -2: "Clean loss",
}


def probe_for_wk(args):
    """Probe all legal KBBKN positions with a given white king square."""
    wk, syzygy_path, scout = args
    tb = chess.syzygy.open_tablebase(syzygy_path)
    counts = {2: 0, 1: 0, 0: 0, -1: 0, -2: 0}
    first_cursed = None

    board = chess.Board(fen=None)

    for bk in range(64):
        if bk == wk:
            continue
        for b1 in range(64):
            if b1 == wk or b1 == bk:
                continue
            for b2 in range(b1 + 1, 64):  # b2 > b1 avoids duplicate bishop pairs
                if b2 == wk or b2 == bk:
                    continue
                for kn in range(64):
                    if kn == wk or kn == bk or kn == b1 or kn == b2:
                        continue

                    board.clear()
                    board.set_piece_at(wk, chess.Piece(chess.KING,   chess.WHITE))
                    board.set_piece_at(bk, chess.Piece(chess.KING,   chess.BLACK))
                    board.set_piece_at(b1, chess.Piece(chess.BISHOP, chess.WHITE))
                    board.set_piece_at(b2, chess.Piece(chess.BISHOP, chess.WHITE))
                    board.set_piece_at(kn, chess.Piece(chess.KNIGHT, chess.BLACK))
                    board.halfmove_clock = 0
                    board.fullmove_number = 1

                    for turn in (chess.WHITE, chess.BLACK):
                        board.turn = turn
                        try:
                            v = tb.probe_wdl(board)
                            counts[v] = counts.get(v, 0) + 1
                            if scout and v in (1, -1) and first_cursed is None:
                                first_cursed = board.fen()
                        except Exception:
                            pass

    tb.close()
    return counts, first_cursed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scout", action="store_true",
                        help="Stop after finding the first cursed position")
    args = parser.parse_args()

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Enumerating all legal KBBKN positions with {n_workers} workers...")
    if args.scout:
        print("SCOUT MODE: will stop after first cursed position found.\n")

    tasks = [(wk, SYZYGY, args.scout) for wk in range(64)]

    totals = {2: 0, 1: 0, 0: 0, -1: 0, -2: 0}
    first_cursed = None

    with mp.Pool(n_workers) as pool:
        for i, (result, fc) in enumerate(
            pool.imap_unordered(probe_for_wk, tasks)
        ):
            for k, v in result.items():
                totals[k] = totals.get(k, 0) + v
            if fc and first_cursed is None:
                first_cursed = fc

            cursed_so_far = totals.get(1, 0) + totals.get(-1, 0)
            total_so_far = sum(totals.values())
            print(f"  wk chunk {i+1}/64 done — "
                  f"total: {total_so_far:,}  cursed: {cursed_so_far:,}", flush=True)

            if args.scout and first_cursed:
                pool.terminate()
                break

    total = sum(totals.values())
    cursed = totals.get(1, 0) + totals.get(-1, 0)

    print(f"\n=== KBBKN Raw WDL Distribution ===")
    for v in [2, 1, 0, -1, -2]:
        n = totals.get(v, 0)
        pct = n / total * 100 if total > 0 else 0
        print(f"  {v:+d}  {LABELS[v]:38s}: {n:>10,}  ({pct:.4f}%)")

    print(f"\nTotal positions counted : {total:,}")
    print(f"Cursed/blessed          : {cursed:,}")
    print(f"Population base rate    : {cursed / total * 100:.4f}%"
          f"  (1 in {int(total / cursed) if cursed else 'inf'})")

    if first_cursed:
        print(f"\nFirst cursed position found: {first_cursed}")

    if cursed == 0:
        print("\nWARNING: Zero cursed positions found — unexpected for KBBKN.")
    else:
        print(f"\nPositive control CONFIRMED: cursed positions exist in KBBKN.")


if __name__ == "__main__":
    main()
