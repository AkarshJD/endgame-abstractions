"""Tests for the counterfactual-board retrograde solver (src/endgame/retro)."""

import random

import numpy as np
import pytest

from endgame.retro import Rules, solve
from endgame.retro.solver import ROOK_DIRS, _geometry


def sq(name: str) -> tuple:
    """'a8' -> (0, 7)."""
    return ord(name[0]) - ord("a"), int(name[1:]) - 1


# ---- known positions on the standard board ---------------------------------


def test_queen_mate_is_lost_for_black_at_depth_zero():
    t = solve("Q")
    wk, bk, q = sq("b6"), sq("a8"), sq("b7")
    assert t.wdl(wk, bk, q, white_to_move=False) == -2
    assert t.depth_b[t.index(wk, bk, q)] == 0


def test_queen_stalemate_is_a_draw():
    t = solve("Q")
    assert t.wdl(sq("b6"), sq("a8"), sq("c7"), white_to_move=False) == 0


def test_black_captures_unprotected_queen():
    t = solve("Q")
    assert t.wdl(sq("h1"), sq("a8"), sq("b7"), white_to_move=False) == 0


def test_black_in_check_with_white_to_move_is_illegal():
    t = solve("R")
    assert t.wdl(sq("e1"), sq("a8"), sq("a1"), white_to_move=True) is None


def test_rook_pawn_with_defending_king_in_corner_is_drawn():
    # Classic a-pawn draw: the black king reaches the promotion corner.
    t = solve("P")
    assert t.wdl(sq("c4"), sq("a8"), sq("a5"), white_to_move=True) == 0


def test_kpk_opposition_decides_the_result():
    # Ke5 / Ke7, pawn e4: whoever must move loses the opposition.
    # White to move only draws; Black to move loses (checked against Syzygy).
    t = solve("P")
    wk, bk, p = sq("e5"), sq("e7"), sq("e4")
    assert t.wdl(wk, bk, p, white_to_move=True) == 0
    assert t.wdl(wk, bk, p, white_to_move=False) == -2


def test_kpk_king_on_key_square_wins_whoever_moves():
    # Ke6 / Ke8, pawn e5: the white king already stands on a key square.
    t = solve("P")
    wk, bk, p = sq("e6"), sq("e8"), sq("e5")
    assert t.wdl(wk, bk, p, white_to_move=True) == 2
    assert t.wdl(wk, bk, p, white_to_move=False) == -2


def test_standard_board_matches_known_theory():
    assert solve("Q").stats["max_depth_plies"] == 20  # KQK: mate in 10
    assert solve("R").stats["max_depth_plies"] == 32  # KRK: mate in 16
    kpk = solve("P").stats
    assert kpk["legal_w"] + kpk["legal_b"] == 331_352  # same count as the Syzygy dataset


# ---- symmetry: board transforms that must not change any result ------------


def _transform_index(w, h, fx, fy, transpose):
    """State index after mirroring x/y and/or transposing, for every state."""
    g = _geometry(w, h)
    nw = h if transpose else w
    n = w * h

    def move(x, y):
        x = (w - 1 - x) if fx else x
        y = (h - 1 - y) if fy else y
        if transpose:
            x, y = y, x
        return y.astype(np.int64) * nw + x

    return (move(g.wkx, g.wky) * n + move(g.bkx, g.bky)) * n + move(g.px, g.py)


@pytest.mark.parametrize("piece", ["Q", "R"])
@pytest.mark.parametrize("w,h", [(6, 6), (5, 7)])
@pytest.mark.parametrize("fx,fy,transpose", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_piece_tables_invariant_under_board_symmetries(piece, w, h, fx, fy, transpose):
    a = solve(piece, Rules(width=w, height=h))
    b = solve(piece, Rules(width=h, height=w)) if transpose else a
    idx = _transform_index(w, h, fx, fy, transpose)
    for name in ("legal_w", "legal_b", "win_w", "lost_b", "depth_w", "depth_b"):
        assert np.array_equal(getattr(a, name), getattr(b, name)[idx]), name


@pytest.mark.parametrize("kings", ["standard", "orthogonal"])
def test_pawn_table_invariant_under_left_right_mirror(kings):
    rules = Rules(width=7, height=7, king_steps=ROOK_DIRS) if kings == "orthogonal" else Rules(7, 7)
    t = solve("P", rules)
    idx = _transform_index(7, 7, 1, 0, 0)
    assert np.array_equal(t.win_w, t.win_w[idx])
    assert np.array_equal(t.lost_b, t.lost_b[idx])


def test_rules_reject_asymmetric_king_steps():
    with pytest.raises(ValueError):
        Rules(king_steps=((1, 0),))


# ---- spot-check against Syzygy (full check: scripts/retro/validate_syzygy.py)


def test_random_positions_match_syzygy():
    chess = pytest.importorskip("chess")
    import chess.syzygy

    from configs.paths import SYZYGY_PATH

    if not (SYZYGY_PATH / "KPvK.rtbw").exists():
        pytest.skip("Syzygy files not present")
    types = {"Q": chess.QUEEN, "R": chess.ROOK, "P": chess.PAWN}
    rng = random.Random(0)
    with chess.syzygy.open_tablebase(str(SYZYGY_PATH)) as tb:
        for piece, ptype in types.items():
            t = solve(piece)
            checked = 0
            while checked < 2000:
                wk, bk, p = rng.sample(range(64), 3)
                stm = rng.random() < 0.5
                board = chess.Board(None)
                board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                board.set_piece_at(p, chess.Piece(ptype, chess.WHITE))
                board.turn = stm
                xy = lambda s: (chess.square_file(s), chess.square_rank(s))  # noqa: E731
                ours = t.wdl(xy(wk), xy(bk), xy(p), stm)
                assert (ours is not None) == board.is_valid(), board.fen()
                if ours is None:
                    continue
                assert ours == tb.probe_wdl(board), board.fen()
                checked += 1
