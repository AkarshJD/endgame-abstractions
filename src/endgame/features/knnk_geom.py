"""
Geometric features for KNNK (King + 2 Knights vs King).

Key concepts:
- KNNK is a theoretical draw: white cannot force mate with best play
- Winning positions only arise as zugzwang when it is BLACK to move
- Both knights must coordinate to cover all escape squares of BK
- Checkmate always occurs in a corner; BK must be driven to an edge first
- Knight pair symmetry: features are order-invariant (sorted by distance to BK)
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


def knight_attacks(sq):
    """Return set of squares attacked by a knight on sq."""
    r, f = chess.square_rank(sq), chess.square_file(sq)
    attacked = set()
    for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nr, nf = r + dr, f + df
        if 0 <= nr <= 7 and 0 <= nf <= 7:
            attacked.add(chess.square(nf, nr))
    return attacked


CORNERS = [chess.A1, chess.A8, chess.H1, chess.H8]


def compute_knnk_features(board: chess.Board):
    wk  = board.king(chess.WHITE)
    bk  = board.king(chess.BLACK)

    knights = list(board.pieces(chess.KNIGHT, chess.WHITE))
    # Always sort by distance to BK so features are order-invariant
    knights.sort(key=lambda sq: chebyshev(sq, bk))
    wn_near, wn_far = knights[0], knights[1]

    bk_r, bk_f = chess.square_rank(bk), chess.square_file(bk)

    features = {}

    # ── BK confinement ────────────────────────────────────────────────────────
    features["bk_edge"]    = dist_to_edge(bk)
    features["bk_on_edge"] = int(dist_to_edge(bk) == 0)
    features["bk_rank"]    = bk_r
    features["bk_file"]    = bk_f
    features["bk_to_corner"] = min(chebyshev(bk, c) for c in CORNERS)
    features["bk_in_corner"] = int(features["bk_to_corner"] == 0)

    # ── WK positioning ────────────────────────────────────────────────────────
    features["wk_edge"]       = dist_to_edge(wk)
    features["d_wk_bk"]       = chebyshev(wk, bk)
    features["m_wk_bk"]       = manhattan(wk, bk)
    features["wk_bk_rank_diff"] = chess.square_rank(wk) - bk_r
    features["wk_bk_file_diff"] = chess.square_file(wk) - bk_f

    # King opposition
    d = chebyshev(wk, bk)
    same_file = chess.square_file(wk) == chess.square_file(bk)
    same_rank  = chess.square_rank(wk) == chess.square_rank(bk)
    features["king_opposition"] = int(d == 2 and (same_file or same_rank))

    # ── Knight distances (order-invariant: near/far relative to BK) ──────────
    features["d_wn_near_bk"] = chebyshev(wn_near, bk)
    features["d_wn_far_bk"]  = chebyshev(wn_far,  bk)
    features["m_wn_near_bk"] = manhattan(wn_near, bk)
    features["m_wn_far_bk"]  = manhattan(wn_far,  bk)
    features["d_nn"]          = chebyshev(wn_near, wn_far)   # inter-knight distance
    features["d_wk_wn_near"]  = chebyshev(wk, wn_near)
    features["d_wk_wn_far"]   = chebyshev(wk, wn_far)

    # Sum / min / max of knight distances to BK
    features["d_wn_sum"]  = features["d_wn_near_bk"] + features["d_wn_far_bk"]
    features["d_wn_min"]  = features["d_wn_near_bk"]   # already sorted
    features["d_wn_max"]  = features["d_wn_far_bk"]

    # ── Knight attack coverage ────────────────────────────────────────────────
    atk_near = knight_attacks(wn_near)
    atk_far  = knight_attacks(wn_far)
    atk_union = atk_near | atk_far

    # BK neighbor squares
    bk_neighbors = set()
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            nr, nf = bk_r + dr, bk_f + df
            if 0 <= nr <= 7 and 0 <= nf <= 7:
                bk_neighbors.add(chess.square(nf, nr))

    features["bk_neighbors_covered_by_near"] = len(atk_near  & bk_neighbors)
    features["bk_neighbors_covered_by_far"]  = len(atk_far   & bk_neighbors)
    features["bk_neighbors_covered_union"]   = len(atk_union & bk_neighbors)
    features["bk_neighbors_covered_both"]    = len(atk_near & atk_far & bk_neighbors)
    features["bk_geo_mobility"]              = len(bk_neighbors)

    # Is BK directly attacked by either knight?
    features["near_attacks_bk"] = int(bk in atk_near)
    features["far_attacks_bk"]  = int(bk in atk_far)
    features["either_attacks_bk"] = int(bk in atk_union)

    # Total union coverage (knight pair board control)
    features["nn_coverage"] = len(atk_union)
    features["nn_overlap"]  = len(atk_near & atk_far)  # squares both knights attack

    # ── Parity / colour ──────────────────────────────────────────────────────
    features["bk_square_color"]    = square_color(bk)
    features["wk_square_color"]    = square_color(wk)
    features["wn_near_color"]      = square_color(wn_near)
    features["wn_far_color"]       = square_color(wn_far)
    features["knights_same_color"] = int(square_color(wn_near) == square_color(wn_far))
    features["kings_same_color"]   = int(square_color(wk) == square_color(bk))

    # ── WK cutting off BK escape ──────────────────────────────────────────────
    features["wk_ahead_of_bk"] = int(chess.square_rank(wk) > bk_r)
    features["wk_right_of_bk"] = int(chess.square_file(wk) > bk_f)

    # ── BK mobility (legal moves when it is black to move) ───────────────────
    if board.turn == chess.BLACK:
        features["bk_mobility"] = sum(
            1 for m in board.generate_legal_moves() if m.from_square == bk
        )
    else:
        # Flip turn temporarily to count black's legal moves
        tmp = chess.Board(board.fen())
        tmp.turn = chess.BLACK
        tmp.clear_stack()
        try:
            features["bk_mobility"] = sum(
                1 for m in tmp.generate_legal_moves() if m.from_square == bk
            )
        except Exception:
            features["bk_mobility"] = features["bk_geo_mobility"]

    # ── Turn ──────────────────────────────────────────────────────────────────
    features["turn"] = int(board.turn == chess.WHITE)

    return features
