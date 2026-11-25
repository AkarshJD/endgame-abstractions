"""
Solve K+X vs K on other board sizes and with other king moves.

Usage:
    python scripts/retro/solve_boards.py
    python scripts/retro/solve_boards.py --sizes 6x6 8x8 10x10 --pieces P --kings standard
    python scripts/retro/solve_boards.py --kings orthogonal

Writes data/processed/retro/<kings>/<W>x<H>/K<X>K.npz (gitignored) and a
summary table to data/processed/retro/summary.csv.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from configs.paths import PROCESSED_DIR
from endgame.retro import Rules, solve
from endgame.retro.solver import KING_STEPS, ROOK_DIRS

KING_VARIANTS = {
    "standard": KING_STEPS,  # chess king: 8 neighbours
    "orthogonal": ROOK_DIRS,  # counterfactual: kings move only orthogonally
}

DEFAULT_SIZES = ["5x5", "6x6", "7x7", "8x8", "9x9", "10x10", "7x10", "10x7"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES, help="WxH, W = files, H = ranks")
    ap.add_argument("--pieces", default="QRP")
    ap.add_argument("--kings", nargs="+", default=list(KING_VARIANTS), choices=list(KING_VARIANTS))
    args = ap.parse_args()

    root = PROCESSED_DIR / "retro"
    rows = []
    for kings in args.kings:
        for size in args.sizes:
            w, h = map(int, size.lower().split("x"))
            rules = Rules(width=w, height=h, king_steps=KING_VARIANTS[kings])
            out = root / kings / f"{w}x{h}"
            out.mkdir(parents=True, exist_ok=True)
            for piece in args.pieces:
                t0 = time.time()
                table = solve(piece, rules)
                table.save(out / f"K{piece}K.npz")
                s = table.stats
                row = {
                    "kings": kings,
                    "board": f"{w}x{h}",
                    "endgame": f"K{piece}K",
                    "positions": s["legal_w"] + s["legal_b"],
                    "white_to_move_win_pct": round(100 * s["win_w"] / max(s["legal_w"], 1), 2),
                    "black_to_move_loss_pct": round(100 * s["lost_b"] / max(s["legal_b"], 1), 2),
                    "max_depth_plies": s["max_depth_plies"],
                    "mates": s["mates"],
                    "stalemates": s["stalemates"],
                    "seconds": round(time.time() - t0, 1),
                }
                rows.append(row)
                print(
                    f"{kings:>10} {row['board']:>6} {row['endgame']}: {row['positions']:>10,} positions, "
                    f"WTM win {row['white_to_move_win_pct']:6.2f}%, BTM loss {row['black_to_move_loss_pct']:6.2f}%, "
                    f"max depth {row['max_depth_plies']:>3} plies ({row['seconds']}s)"
                )

    summary = pd.DataFrame(rows)
    path = root / "summary.csv"
    if path.exists():  # keep rows for configurations not re-run this time
        old = pd.read_csv(path)
        keys = ["kings", "board", "endgame"]
        old = old[~old.set_index(keys).index.isin(summary.set_index(keys).index)]
        summary = pd.concat([old, summary], ignore_index=True)
    summary.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
