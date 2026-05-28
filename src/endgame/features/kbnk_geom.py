"""
Geometric features for KBNK (King + Bishop + Knight vs King).

Key concepts this feature set encodes:
- BK must be driven to a corner matching the bishop's square colour
- Knight + Bishop must coordinate to cut off escape routes
- The "wrong corner" trap (BK in corner of wrong colour = stalemate risk)
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
    """0 = dark, 1 = light (same convention as bishop colour)."""
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2


def corner_color(corner):
    """Color of a corner square."""
    return square_color(corner)


CORNERS = [chess.A1, chess.A8, chess.H1, chess.H8]


def compute_kbnk_features(board: chess.Board):
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)

    bishop = None
    knight = None
    for sq in board.pieces(chess.BISHOP, chess.WHITE):
        bishop = sq
    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
        knight = sq

    features = {}

    # ── Bishop colour ─────────────────────────────────────────────────────────
    bishop_color = square_color(bishop)       # 0=dark, 1=light
    features["bishop_color"] = bishop_color

    # ── Corner distances ──────────────────────────────────────────────────────
    correct_corners = [c for c in CORNERS if corner_color(c) == bishop_color]
    wrong_corners   = [c for c in CORNERS if corner_color(c) != bishop_color]

    bk_to_correct_corner = min(chebyshev(bk, c) for c in correct_corners)
    bk_to_wrong_corner   = min(chebyshev(bk, c) for c in wrong_corners)
    wk_to_correct_corner = min(chebyshev(wk, c) for c in correct_corners)

    features["bk_to_correct_corner"] = bk_to_correct_corner
    features["bk_to_wrong_corner"]   = bk_to_wrong_corner
    features["wk_to_correct_corner"] = wk_to_correct_corner
    features["corner_color_advantage"] = bk_to_wrong_corner - bk_to_correct_corner

    # ── Is BK near/in a wrong-colour corner (stalemate trap) ─────────────────
    features["bk_in_wrong_corner"] = int(bk_to_wrong_corner == 0)
    features["bk_near_wrong_corner"] = int(bk_to_wrong_corner <= 1)
    features["bk_in_correct_corner"] = int(bk_to_correct_corner == 0)

    # ── Edge and centre proximity ─────────────────────────────────────────────
    features["bk_edge"] = dist_to_edge(bk)
    features["wk_edge"] = dist_to_edge(wk)
    features["bk_on_edge"] = int(dist_to_edge(bk) == 0)

    bk_r, bk_f = chess.square_rank(bk), chess.square_file(bk)
    features["bk_rank"] = bk_r
    features["bk_file"] = bk_f

    # ── Inter-piece distances ─────────────────────────────────────────────────
    features["d_wk_bk"]  = chebyshev(wk, bk)
    features["d_wn_bk"]  = chebyshev(knight, bk)
    features["d_wb_bk"]  = chebyshev(bishop, bk)
    features["d_wk_wn"]  = chebyshev(wk, knight)
    features["d_wk_wb"]  = chebyshev(wk, bishop)
    features["d_wn_wb"]  = chebyshev(knight, bishop)

    features["m_wk_bk"]  = manhattan(wk, bk)
    features["m_wn_bk"]  = manhattan(knight, bk)
    features["m_wb_bk"]  = manhattan(bishop, bk)

    # ── Knight proximity: how close is the knight to cutting off the BK ───────
    features["knight_near_bk"] = int(chebyshev(knight, bk) <= 2)

    # ── Bishop diagonal alignment with BK ─────────────────────────────────────
    wb_r, wb_f = chess.square_rank(bishop), chess.square_file(bishop)
    # On same diagonal if |rank_diff| == |file_diff|
    features["bishop_on_bk_diagonal"] = int(
        abs(wb_r - bk_r) == abs(wb_f - chess.square_file(bk))
    )
    features["bishop_attacks_bk_color"] = int(square_color(bishop) == square_color(bk))

    # ── Knight coverage of squares adjacent to BK ────────────────────────────
    knight_r, knight_f = chess.square_rank(knight), chess.square_file(knight)
    knight_attacks = set()
    for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nr, nf = knight_r + dr, knight_f + df
        if 0 <= nr <= 7 and 0 <= nf <= 7:
            knight_attacks.add(chess.square(nf, nr))

    bk_neighbors = set()
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            nr, nf = bk_r + dr, chess.square_file(bk) + df
            if 0 <= nr <= 7 and 0 <= nf <= 7:
                bk_neighbors.add(chess.square(nf, nr))

    features["knight_covers_bk_neighbors"] = len(knight_attacks & bk_neighbors)
    features["knight_attacks_bk"] = int(bk in knight_attacks)

    # ── Parity ────────────────────────────────────────────────────────────────
    features["bk_square_color"] = square_color(bk)
    features["wk_square_color"] = square_color(wk)
    features["kings_same_color"] = int(square_color(wk) == square_color(bk))
    features["bk_same_color_as_bishop"] = int(square_color(bk) == bishop_color)

    # ── Signed offsets between kings ─────────────────────────────────────────
    features["wk_bk_rank_diff"] = chess.square_rank(wk) - bk_r
    features["wk_bk_file_diff"] = chess.square_file(wk) - chess.square_file(bk)

    # ── King opposition ───────────────────────────────────────────────────────
    d = chebyshev(wk, bk)
    same_file = chess.square_file(wk) == chess.square_file(bk)
    same_rank = chess.square_rank(wk) == chess.square_rank(bk)
    features["king_opposition"] = int(d == 2 and (same_file or same_rank))

    # ── WK-BK confinement: how well does WK box in BK ────────────────────────
    features["wk_ahead_of_bk"] = int(chess.square_rank(wk) > bk_r)

    # ── Mobility ──────────────────────────────────────────────────────────────
    features["bk_mobility"] = sum(
        1 for m in board.generate_legal_moves()
        if board.turn == chess.BLACK and m.from_square == bk
    ) if board.turn == chess.BLACK else sum(
        1 for m in chess.Board(board.fen().replace(" w ", " b ", 1)
                               if board.turn == chess.WHITE else board.fen()
                               ).generate_legal_moves()
    )
    # Simpler version: geometric mobility (ignores pins/blocks)
    bk_geo_mobility = 0
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            nr, nf = bk_r + dr, chess.square_file(bk) + df
            if 0 <= nr <= 7 and 0 <= nf <= 7:
                bk_geo_mobility += 1
    features["bk_geo_mobility"] = bk_geo_mobility

    # ── Turn ──────────────────────────────────────────────────────────────────
    features["turn"] = int(board.turn == chess.WHITE)

    return features
