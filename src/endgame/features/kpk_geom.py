import chess


# ---------- Metric geometry ----------

def chebyshev(a, b):
    return max(
        abs(chess.square_file(a) - chess.square_file(b)),
        abs(chess.square_rank(a) - chess.square_rank(b))
    )


def manhattan(a, b):
    return abs(chess.square_file(a) - chess.square_file(b)) + \
           abs(chess.square_rank(a) - chess.square_rank(b))


def euclid_sq(a, b):
    df = chess.square_file(a) - chess.square_file(b)
    dr = chess.square_rank(a) - chess.square_rank(b)
    return df * df + dr * dr


# ---------- Topology / board geometry ----------

def dist_to_edge(sq):
    return min(
        chess.square_file(sq),
        7 - chess.square_file(sq),
        chess.square_rank(sq),
        7 - chess.square_rank(sq),
    )


def king_mobility(sq):
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    count = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            nf, nr = file + df, rank + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                count += 1
    return count


# ---------- Pure geometry (no chess semantics) ----------

def triangle_area2(a, b, c):
    x1, y1 = chess.square_file(a), chess.square_rank(a)
    x2, y2 = chess.square_file(b), chess.square_rank(b)
    x3, y3 = chess.square_file(c), chess.square_rank(c)
    return abs(
        x1 * (y2 - y3) +
        x2 * (y3 - y1) +
        x3 * (y1 - y2)
    )


# ---------- Feature extraction ----------

def extract_kpk_features(board: chess.Board):

    pieces = board.piece_map()

    wk = bk = pawn = None

    for sq, p in pieces.items():
        if p.piece_type == chess.KING:
            if p.color == chess.WHITE:
                wk = sq
            else:
                bk = sq
        elif p.piece_type == chess.PAWN:
            pawn = sq

    pawn_rank = chess.square_rank(pawn)
    promotion_rank = 7

    features = {}

    # --- Existing distances (Chebyshev) ---
    features["d_wk_bk"] = chebyshev(wk, bk)
    features["d_wk_p"]  = chebyshev(wk, pawn)
    features["d_bk_p"]  = chebyshev(bk, pawn)

    # --- Metric variants ---
    features["m_wk_bk"] = manhattan(wk, bk)
    features["m_wk_p"]  = manhattan(wk, pawn)
    features["m_bk_p"]  = manhattan(bk, pawn)

    features["e2_wk_bk"] = euclid_sq(wk, bk)
    features["e2_wk_p"]  = euclid_sq(wk, pawn)
    features["e2_bk_p"]  = euclid_sq(bk, pawn)

    # --- Pawn progress ---
    features["pawn_rank"] = pawn_rank
    features["pawn_rank_norm"] = pawn_rank / 7.0
    features["d_promo"] = promotion_rank - pawn_rank
    features["pawn_file"] = chess.square_file(pawn)

    # --- Signed directional geometry ---
    features["wk_bk_dr"] = chess.square_rank(wk) - chess.square_rank(bk)
    features["wk_bk_df"] = chess.square_file(wk) - chess.square_file(bk)

    features["wk_p_dr"] = chess.square_rank(wk) - pawn_rank
    features["wk_p_df"] = chess.square_file(wk) - chess.square_file(pawn)

    features["bk_p_dr"] = chess.square_rank(bk) - pawn_rank
    features["bk_p_df"] = chess.square_file(bk) - chess.square_file(pawn)

    # --- Absolute gaps (kept for comparison) ---
    features["rank_gap"] = abs(
        chess.square_rank(wk) - chess.square_rank(bk)
    )
    features["file_gap"] = abs(
        chess.square_file(wk) - chess.square_file(bk)
    )

    # --- Edge / boundary effects ---
    features["wk_edge"] = dist_to_edge(wk)
    features["bk_edge"] = dist_to_edge(bk)
    features["p_edge"]  = dist_to_edge(pawn)

    # --- Local mobility (static, one-ply geometry) ---
    features["wk_mobility"] = king_mobility(wk)
    features["bk_mobility"] = king_mobility(bk)

    # --- Relative ordering (ordinal, not theory) ---
    features["wk_vs_p_rank"] = chess.square_rank(wk) - pawn_rank
    features["bk_vs_p_rank"] = chess.square_rank(bk) - pawn_rank

    # --- Triangle geometry ---
    features["triangle_area2"] = triangle_area2(wk, bk, pawn)

    # --- Tempo ---
    features["turn"] = int(board.turn)

    return features
