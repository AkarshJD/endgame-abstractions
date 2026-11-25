"""
Check the retrograde solver against Syzygy on the standard 8x8 board.

For every placement of (white king, black king, piece) on distinct squares,
with each side to move, this compares
  1. legality: our solver's legal set vs python-chess Board.is_valid()
  2. outcome:  our WDL vs Syzygy probe_wdl, on every position both accept

Usage:
    python scripts/retro/validate_syzygy.py            # KQK, KRK, KPK
    python scripts/retro/validate_syzygy.py --pieces R

Writes logs/retro_validation/<timestamp>/summary.json (+ mismatches.csv if any).
Exit code is non-zero if any outcome disagrees.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.syzygy

from configs.paths import PROJECT_ROOT, SYZYGY_PATH
from endgame.retro import Rules, solve

PIECE_TYPES = {"Q": chess.QUEEN, "R": chess.ROOK, "P": chess.PAWN, "B": chess.BISHOP, "N": chess.KNIGHT}

_TABLE = None
_TB = None


def _init(piece: str) -> None:
    global _TABLE, _TB
    _TABLE = solve(piece, Rules())
    _TB = chess.syzygy.open_tablebase(str(SYZYGY_PATH))


def _check_wk(args):
    """Compare every position with the white king on square `wk`."""
    piece, wk = args
    t = _TABLE
    board = chess.Board(None)
    counts = {"both_legal": 0, "ours_only": 0, "chess_only": 0, "agree": 0, "cursed": 0}
    mismatches = []
    for bk in chess.SQUARES:
        if bk == wk:
            continue
        for p in chess.SQUARES:
            if p in (wk, bk):
                continue
            board.clear()
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(p, chess.Piece(PIECE_TYPES[piece], chess.WHITE))
            for white_to_move in (True, False):
                board.turn = white_to_move
                ours = t.wdl(
                    (chess.square_file(wk), chess.square_rank(wk)),
                    (chess.square_file(bk), chess.square_rank(bk)),
                    (chess.square_file(p), chess.square_rank(p)),
                    white_to_move,
                )
                valid = board.is_valid()
                if ours is None and not valid:
                    continue
                if ours is None:
                    counts["chess_only"] += 1
                    mismatches.append((board.fen(), "legality", "illegal", "valid"))
                    continue
                if not valid:
                    counts["ours_only"] += 1
                    mismatches.append((board.fen(), "legality", "legal", "invalid"))
                    continue
                counts["both_legal"] += 1
                syz = _TB.probe_wdl(board)
                if syz in (1, -1):
                    counts["cursed"] += 1
                if ours == syz:
                    counts["agree"] += 1
                else:
                    mismatches.append((board.fen(), "wdl", ours, syz))
    return counts, mismatches


def validate(piece: str, workers: int = 7) -> dict:
    t0 = time.time()
    table = solve(piece, Rules())
    with Pool(workers, initializer=_init, initargs=(piece,)) as pool:
        results = pool.map(_check_wk, [(piece, wk) for wk in chess.SQUARES])
    totals = {k: sum(r[0][k] for r in results) for k in results[0][0]}
    mismatches = [m for r in results for m in r[1]]
    totals.update(
        endgame=f"K{piece}K",
        wdl_mismatches=sum(1 for m in mismatches if m[1] == "wdl"),
        legality_mismatches=sum(1 for m in mismatches if m[1] == "legality"),
        solver_stats=table.stats,
        seconds=round(time.time() - t0, 1),
    )
    return totals, mismatches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="QRP", help="which K+X vs K tables to check, e.g. QRP")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    out = PROJECT_ROOT / "logs" / "retro_validation" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True, exist_ok=True)
    summary, all_mismatches = {}, []
    for piece in args.pieces:
        totals, mismatches = validate(piece, args.workers)
        summary[f"K{piece}K"] = totals
        all_mismatches += [(f"K{piece}K", *m) for m in mismatches]
        print(
            f"K{piece}K: {totals['agree']:,}/{totals['both_legal']:,} WDL agree, "
            f"{totals['wdl_mismatches']} WDL mismatches, "
            f"{totals['legality_mismatches']} legality mismatches "
            f"(ours-only {totals['ours_only']}, chess-only {totals['chess_only']}), "
            f"cursed {totals['cursed']}, {totals['seconds']}s"
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if all_mismatches:
        with open(out / "mismatches.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["endgame", "fen", "kind", "ours", "syzygy_or_chess"])
            w.writerows(all_mismatches)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")
    return 1 if any(v["wdl_mismatches"] for v in summary.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
