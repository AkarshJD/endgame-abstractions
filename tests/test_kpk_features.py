"""KPK feature extraction returns every feature, zeros included."""

import chess

from endgame.features.kpk_geom import extract_kpk_features


def test_black_to_move_has_turn_zero():
    feats = extract_kpk_features(chess.Board("8/4k3/8/4K3/4P3/8/8/8 b - - 0 1"))
    assert feats["turn"] == 0


def test_every_position_has_the_same_features():
    a = extract_kpk_features(chess.Board("8/4k3/8/4K3/4P3/8/8/8 b - - 0 1"))
    b = extract_kpk_features(chess.Board("k7/8/8/8/8/8/P7/7K w - - 0 1"))
    assert a.keys() == b.keys()
    assert a["pawn_file"] == 4 and b["pawn_file"] == 0  # a-file is 0, not missing
    assert a["wk_bk_df"] == 0  # kings on the same file
