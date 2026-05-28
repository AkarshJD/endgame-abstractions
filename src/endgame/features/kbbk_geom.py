"""
Geometric features for KBBK (King + 2 Bishops vs King).

Key concepts:
- Same-color bishops: theoretical draw (only control one square color)
- Diff-color bishops: forced win; BK driven to any corner (no wrong corner)
- The light bishop and dark bishop are distinguished by square color,
  not by board position — features are labeled light_b / dark_b accordingly
- When bishops share a color, both are treated as "light_b" by convention
  and the same_color flag is set
"""

import chess


def chebyshev(a, b):
    return max(
        abs(chess.square_file(a) - chess.square_file(b)),
        abs(chess.square_rank(a) - chess.square_rank(b)),
    )


def manhattan(a, b):
    return (
        abs(chess.square_file(a) - chess.square_file(b))
        + abs(chess.square_rank(a) - chess.square_rank(b))
    )


def dist_to_edge(sq):
    return min(
        chess.square_file(sq),
        7 - chess.square_file(sq),
        chess.square_rank(sq),
        7 - chess.square_rank(sq),
    )


def square_color(sq):
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2


CORNERS = [chess.A1, chess.A8, chess.H1, chess.H8]


def bishop_attacks(sq):
    """All squares on both diagonals through sq (excluding sq itself)."""
    attacked = set()
    r, f = chess.square_rank(sq), chess.square_file(sq)
    for dr, df in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        nr, nf = r + dr, f + df
        while 0 <= nr <= 7 and 0 <= nf <= 7:
            attacked.add(chess.square(nf, nr))
            nr += dr
            nf += df
    return attacked


def compute_kbbk_features(board: chess.Board):
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)

    bishops = list(board.pieces(chess.BISHOP, chess.WHITE))

    b_colors = [square_color(b) for b in bishops]
    same_color = int(b_colors[0] == b_colors[1])

    if same_color:
        # Sort by distance to BK for order-invariance
        bishops.sort(key=lambda sq: chebyshev(sq, bk))
        light_b, dark_b = bishops[0], bishops[1]
    else:
        # Assign by actual square color
        if square_color(bishops[0]) == 1:
            light_b, dark_b = bishops[0], bishops[1]
        else:
            light_b, dark_b = bishops[1], bishops[0]

    bk_r, bk_f = chess.square_rank(bk), chess.square_file(bk)

    features = {}

    # ── Bishop colour structure ───────────────────────────────────────────────
    features["bishops_same_color"] = same_color
    features["light_b_color"]      = square_color(light_b)   # sanity: always 1 if diff-color
    features["dark_b_color"]       = square_color(dark_b)    # sanity: always 0 if diff-color

    # ── BK confinement ────────────────────────────────────────────────────────
    features["bk_edge"]      = dist_to_edge(bk)
    features["bk_on_edge"]   = int(dist_to_edge(bk) == 0)
    features["bk_rank"]      = bk_r
    features["bk_file"]      = bk_f
    features["bk_to_corner"] = min(chebyshev(bk, c) for c in CORNERS)
    features["bk_in_corner"] = int(features["bk_to_corner"] == 0)

    # BK neighbor squares (geometric mobility)
    bk_neighbors = set()
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            nr, nf = bk_r + dr, bk_f + df
            if 0 <= nr <= 7 and 0 <= nf <= 7:
                bk_neighbors.add(chess.square(nf, nr))
    features["bk_geo_mobility"] = len(bk_neighbors)

    # ── WK positioning ────────────────────────────────────────────────────────
    features["wk_edge"]          = dist_to_edge(wk)
    features["d_wk_bk"]          = chebyshev(wk, bk)
    features["m_wk_bk"]          = manhattan(wk, bk)
    features["wk_bk_rank_diff"]  = chess.square_rank(wk) - bk_r
    features["wk_bk_file_diff"]  = chess.square_file(wk) - bk_f

    d = chebyshev(wk, bk)
    same_file = chess.square_file(wk) == chess.square_file(bk)
    same_rank  = chess.square_rank(wk) == chess.square_rank(bk)
    features["king_opposition"] = int(d == 2 and (same_file or same_rank))
    features["wk_ahead_of_bk"]  = int(chess.square_rank(wk) > bk_r)

    # ── Bishop distances ──────────────────────────────────────────────────────
    features["d_light_b_bk"]  = chebyshev(light_b, bk)
    features["d_dark_b_bk"]   = chebyshev(dark_b,  bk)
    features["m_light_b_bk"]  = manhattan(light_b, bk)
    features["m_dark_b_bk"]   = manhattan(dark_b,  bk)
    features["d_bb"]           = chebyshev(light_b, dark_b)
    features["d_wk_light_b"]   = chebyshev(wk, light_b)
    features["d_wk_dark_b"]    = chebyshev(wk, dark_b)
    features["d_b_sum"]        = features["d_light_b_bk"] + features["d_dark_b_bk"]
    features["d_b_min"]        = min(features["d_light_b_bk"], features["d_dark_b_bk"])
    features["d_b_max"]        = max(features["d_light_b_bk"], features["d_dark_b_bk"])

    # ── Bishop diagonal alignment with BK ────────────────────────────────────
    for name, sq in [("light_b", light_b), ("dark_b", dark_b)]:
        b_r, b_f = chess.square_rank(sq), chess.square_file(sq)
        features[f"{name}_on_bk_diagonal"] = int(
            abs(b_r - bk_r) == abs(b_f - bk_f)
        )

    # ── Bishop attack coverage of BK neighbours ───────────────────────────────
    atk_light = bishop_attacks(light_b)
    atk_dark  = bishop_attacks(dark_b)
    atk_union = atk_light | atk_dark

    features["bk_neighbors_covered_light"] = len(atk_light & bk_neighbors)
    features["bk_neighbors_covered_dark"]  = len(atk_dark  & bk_neighbors)
    features["bk_neighbors_covered_union"] = len(atk_union & bk_neighbors)
    features["bk_attacked_by_light"]       = int(bk in atk_light)
    features["bk_attacked_by_dark"]        = int(bk in atk_dark)
    features["bb_coverage"]                = len(atk_union)

    # ── Parity ────────────────────────────────────────────────────────────────
    features["bk_square_color"]  = square_color(bk)
    features["wk_square_color"]  = square_color(wk)
    features["kings_same_color"] = int(square_color(wk) == square_color(bk))

    # ── Turn ──────────────────────────────────────────────────────────────────
    features["turn"] = int(board.turn == chess.WHITE)

    return features
