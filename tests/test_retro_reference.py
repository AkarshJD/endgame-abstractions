"""Cross-check the retrograde solver against the independent reference solver.

Syzygy only covers 8x8 chess (tests/test_retro.py). These tests cover other
board sizes and orthogonal-only kings. Full sweep:
scripts/retro/validate_reference.py.
"""

import ast
from pathlib import Path

import pytest

from endgame.retro import Rules
from endgame.retro.crosscheck import crosscheck
from endgame.retro.reference import solve_reference
from endgame.retro.solver import KING_STEPS, ROOK_DIRS

KINGS = {"standard": KING_STEPS, "orthogonal": ROOK_DIRS}


def test_reference_shares_no_code_with_the_solver():
    """The check is only independent if reference.py never imports solver.py."""
    tree = ast.parse(Path(__file__).parents[1].joinpath("src/endgame/retro/reference.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {a.name for a in node.names}
    assert not any("solver" in name or "endgame" in name for name in imported), imported


# 5x5 is square; 4x6 and 6x4 catch width/height mix-ups, and on 6x4 the pawn
# double step is switched off (it would land on the last rank).
@pytest.mark.parametrize("kings", list(KINGS))
@pytest.mark.parametrize("w,h", [(5, 5), (4, 6), (6, 4)])
@pytest.mark.parametrize("piece", list("QRBNP"))
def test_solver_matches_reference_on_every_position(piece, w, h, kings):
    r = crosscheck(piece, Rules(width=w, height=h, king_steps=KINGS[kings]))
    assert r["positions"] > 0
    assert r["legality_mismatches"] == 0, r["examples"]
    assert r["wdl_mismatches"] == 0, r["examples"]
    assert r["depth_mismatches"] == 0, r["examples"]


def sq(name: str) -> tuple:
    """'a5' -> (0, 4)."""
    return ord(name[0]) - ord("a"), int(name[1:]) - 1


def test_reference_orthogonal_king_mate_and_stalemate_by_hand():
    # 5x5, orthogonal kings, Black to move. Kc5 vs Ka5: two squares apart, legal.
    # The black king's only squares are a4 and b5; Kc5 covers b5.
    ref = solve_reference("R", 5, 5, ROOK_DIRS)
    wk, bk = sq("c5"), sq("a5")
    # Ra1 checks along the a-file and covers a4: mate.
    assert ref[(wk, bk, sq("a1"), False)] == (-2, 0)
    # Re4 covers a4 along the 4th rank without giving check: stalemate.
    assert ref[(wk, bk, sq("e4"), False)] == (0, -1)
    # Kc4 instead: a chess king covers b5 (diagonally) and b4, so Ra1 mates;
    # an orthogonal king covers b4 but not b5, so Black escapes.
    chess = solve_reference("R", 5, 5, KING_STEPS)
    pos = (sq("c4"), bk, sq("a1"), False)
    assert chess[pos] == (-2, 0)
    assert ref[pos] != (-2, 0)
