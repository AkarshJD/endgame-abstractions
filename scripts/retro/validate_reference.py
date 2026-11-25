"""
Check the retrograde solver against the independent reference solver.

This covers what Syzygy cannot: other board sizes and other king moves. Every
legal position is compared for legality, WDL and depth. See
src/endgame/retro/reference.py for how the reference differs from the solver.

Usage:
    python scripts/retro/validate_reference.py                 # default sweep
    python scripts/retro/validate_reference.py --sizes 9x9 --pieces P --kings orthogonal

Writes logs/retro_validation/<timestamp>_reference/summary.json.
Exit code is non-zero if anything disagrees.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from multiprocessing import Pool

from configs.paths import PROJECT_ROOT
from endgame.retro import Rules
from endgame.retro.crosscheck import crosscheck
from endgame.retro.reference import solve_reference
from endgame.retro.solver import KING_STEPS, ROOK_DIRS

KING_VARIANTS = {"standard": KING_STEPS, "orthogonal": ROOK_DIRS}
DEFAULT_SIZES = ["5x5", "6x6", "7x7", "8x8", "7x10", "10x7"]


def _check_board(args):
    """One worker task: every requested piece on one board, so the pawn table
    reuses the Q and R reference tables it promotes into."""
    kings, size, pieces = args
    w, h = map(int, size.lower().split("x"))
    rules = Rules(width=w, height=h, king_steps=KING_VARIANTS[kings])
    rows = []
    for piece in pieces:
        t0 = time.time()
        r = crosscheck(piece, rules)
        r.update(kings=kings, seconds=round(time.time() - t0, 1))
        rows.append(r)
    solve_reference.cache_clear()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES, help="WxH, W = files, H = ranks")
    ap.add_argument("--pieces", default="QRP")
    ap.add_argument("--kings", nargs="+", default=list(KING_VARIANTS), choices=list(KING_VARIANTS))
    ap.add_argument("--workers", type=int, default=4, help="a 7x10 worker peaks at about 1.2 GB")
    args = ap.parse_args()

    tasks = [(k, s, args.pieces) for k in args.kings for s in args.sizes]
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = [r for batch in pool.imap(_check_board, tasks) for r in batch]

    failed = 0
    for r in rows:
        bad = r["legality_mismatches"] + r["wdl_mismatches"] + r["depth_mismatches"]
        failed += bad > 0
        print(
            f"{r['kings']:>10} {r['board']:>5} {r['endgame']}: {r['agree']:>9,}/{r['positions']:>9,} agree, "
            f"mismatches legality {r['legality_mismatches']} wdl {r['wdl_mismatches']} "
            f"depth {r['depth_mismatches']} ({r['seconds']}s)"
        )
    total = sum(r["positions"] for r in rows)
    print(f"{len(rows)} tables, {total:,} positions, {failed} tables with mismatches, {time.time() - t0:.0f}s")

    out = PROJECT_ROOT / "logs" / "retro_validation" / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_reference")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
