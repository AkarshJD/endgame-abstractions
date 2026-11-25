"""Compare solver.py with the independent reference solver, position by position."""

from __future__ import annotations

import numpy as np

from .reference import solve_reference
from .solver import Rules, solve


def crosscheck(piece: str, rules: Rules, max_examples: int = 20) -> dict:
    """Check legality, WDL and depth of every position of K + piece vs K.

    Returns counts plus up to `max_examples` disagreements.
    """
    ref = solve_reference(
        piece, rules.width, rules.height, tuple(rules.king_steps),
        rules.pawn_double_step, tuple(rules.promotions),
    )
    t = solve(piece, rules)
    both_legal = {True: 0, False: 0}
    counts = {"legality": 0, "wdl": 0, "depth": 0}
    examples = []
    for (wk, bk, p, wtm), (wdl, depth) in ref.items():
        ours = t.wdl(wk, bk, p, wtm)
        if ours is None:
            kind, got, want = "legality", "illegal", "legal"
        else:
            both_legal[wtm] += 1
            our_depth = int((t.depth_w if wtm else t.depth_b)[t.index(wk, bk, p)])
            if ours != wdl:
                kind, got, want = "wdl", ours, wdl
            elif our_depth != depth:
                kind, got, want = "depth", our_depth, depth
            else:
                continue
        counts[kind] += 1
        if len(examples) < max_examples:
            examples.append({"wk": wk, "bk": bk, "p": p, "white_to_move": wtm,
                             "kind": kind, "solver": got, "reference": want})

    # Positions the solver calls legal but the reference never produced.
    solver_only = (int(np.sum(t.legal_w)) - both_legal[True]) + (int(np.sum(t.legal_b)) - both_legal[False])
    counts["legality"] += solver_only
    return {
        "endgame": f"K{piece}K",
        "board": f"{rules.width}x{rules.height}",
        "positions": len(ref),
        "agree": sum(both_legal.values()) - counts["wdl"] - counts["depth"],
        "legality_mismatches": counts["legality"],
        "solver_only_legal": solver_only,
        "wdl_mismatches": counts["wdl"],
        "depth_mismatches": counts["depth"],
        "examples": examples,
    }
