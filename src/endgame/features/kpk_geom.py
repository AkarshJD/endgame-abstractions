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


# ---------- KPK pawn-theory helpers (analytic proxies) ----------

def key_squares_for_white_pawn(pawn_sq: int):
    f = chess.square_file(pawn_sq)
    r = chess.square_rank(pawn_sq)
    key_r = min(r + 2, 7)
    out = []
    for df in (-1, 0, 1):
        nf = f + df
        if 0 <= nf <= 7:
            out.append(chess.square(nf, key_r))
    return out


def promotion_square(pawn_sq: int):
    return chess.square(chess.square_file(pawn_sq), 7)


def promotion_square_zone(pawn_sq: int):
    psq = promotion_square(pawn_sq)
    pf, pr = chess.square_file(psq), chess.square_rank(psq)
    zone = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            nf, nr = pf + df, pr + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                zone.append(chess.square(nf, nr))
    return set(zone)


def pawn_one_step_square(pawn_sq: int):
    f = chess.square_file(pawn_sq)
    r = chess.square_rank(pawn_sq)
    if r >= 7:
        return None
    return chess.square(f, r + 1)


def pawn_push_gives_check_after_push(board: chess.Board, pawn_sq: int, bk_sq: int):
    to_sq = pawn_one_step_square(pawn_sq)
    if to_sq is None:
        return 0
    if board.piece_at(to_sq) is not None:
        return 0

    pf = chess.square_file(to_sq)
    pr = chess.square_rank(to_sq)

    attacked = []
    # from new pawn square, pawn attacks (pf-1, pr+1) and (pf+1, pr+1)
    if 0 <= pf - 1 <= 7 and 0 <= pr + 1 <= 7:
        attacked.append(chess.square(pf - 1, pr + 1))
    if 0 <= pf + 1 <= 7 and 0 <= pr + 1 <= 7:
        attacked.append(chess.square(pf + 1, pr + 1))

    return int(bk_sq in attacked)


def legal_move_count_for_turn(board: chess.Board, turn: bool):
    b = board.copy(stack=False)
    b.turn = turn
    return sum(1 for _ in b.legal_moves)


def _keep_nonzero(d):
    """
    Keep only:
      - ints/floats != 0
      - bool True
      - strings non-empty
      - everything else that isn't None
    """
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                out[k] = int(v)
            continue
        if isinstance(v, (int, float)):
            if v != 0:
                out[k] = v
            continue
        if isinstance(v, str):
            if v.strip():
                out[k] = v
            continue
        # fallback: keep non-empty containers or any other object
        try:
            if len(v) > 0:
                out[k] = v
        except Exception:
            out[k] = v
    return out


# ---------- Feature extraction ----------

def extract_kpk_features(board: chess.Board):

    pieces = board.piece_map()
    wk = bk = pawn = None

    for sq, p in pieces.items():
        if p.piece_type == chess.KING:
            wk = sq if p.color == chess.WHITE else wk
            bk = sq if p.color == chess.BLACK else bk
        elif p.piece_type == chess.PAWN and p.color == chess.WHITE:
            pawn = sq

    if wk is None or bk is None or pawn is None:
        raise ValueError("Expected KPK with a WHITE pawn: WK, BK, WP.")

    pawn_rank = chess.square_rank(pawn)
    promotion_rank = 7

    wk_r, wk_f = chess.square_rank(wk), chess.square_file(wk)
    bk_r, bk_f = chess.square_rank(bk), chess.square_file(bk)
    p_f = chess.square_file(pawn)

    rank_gap = abs(wk_r - bk_r)
    file_gap = abs(wk_f - bk_f)
    d_wk_bk = chebyshev(wk, bk)

    wk_mob = king_mobility(wk)
    bk_mob = king_mobility(bk)

    features = {}

    # --- Distances ---
    features["d_wk_bk"] = d_wk_bk
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
    features["pawn_file"] = p_f

    # --- Signed offsets ---
    features["wk_bk_dr"] = wk_r - bk_r
    features["wk_bk_df"] = wk_f - bk_f

    features["wk_p_dr"] = wk_r - pawn_rank
    features["wk_p_df"] = wk_f - p_f

    features["bk_p_dr"] = bk_r - pawn_rank
    features["bk_p_df"] = bk_f - p_f

    # --- Gaps ---
    features["rank_gap"] = rank_gap
    features["file_gap"] = file_gap

    # --- Edge / mobility ---
    features["wk_edge"] = dist_to_edge(wk)
    features["bk_edge"] = dist_to_edge(bk)
    features["p_edge"]  = dist_to_edge(pawn)

    features["wk_mobility"] = wk_mob
    features["bk_mobility"] = bk_mob

    # --- Ordering ---
    features["wk_vs_p_rank"] = wk_r - pawn_rank
    features["bk_vs_p_rank"] = bk_r - pawn_rank

    # --- Triangle geometry ---
    features["triangle_area2"] = triangle_area2(wk, bk, pawn)

    # --- Tempo ---
    features["turn"] = int(board.turn == chess.WHITE)

    # =======================
    # New non-leaky features
    # =======================

    features["rank_opposition_parity"] = int((rank_gap % 2 == 1) and (file_gap == 0))
    features["file_opposition_parity"] = int((file_gap % 2 == 1) and (rank_gap == 0))
    features["direct_opposition"] = int((d_wk_bk == 2) and (rank_gap == 0 or file_gap == 0))
    features["diagonal_opposition"] = int((rank_gap == file_gap) and (d_wk_bk > 1) and ((d_wk_bk - 1) % 2 == 0))

    ks = set(key_squares_for_white_pawn(pawn))
    features["wk_controls_key_square"] = int(wk in ks)

    pzone = promotion_square_zone(pawn)
    features["bk_controls_promotion_square"] = int(bk in pzone)

    features["wk_forward_penetration"] = pawn_rank - wk_r
    features["bk_blockade_zone"] = int(bk_r >= pawn_rank)

    opposition_struct = (
        features["direct_opposition"]
        or features["rank_opposition_parity"]
        or features["file_opposition_parity"]
    )
    features["has_opposition_with_turn"] = int(bool(opposition_struct) ^ bool(features["turn"]))

    features["wk_triangulation_potential"] = int((wk_mob >= 2) and (d_wk_bk == 2))
    features["king_square_color_parity"] = int((wk_r + wk_f) % 2)
    features["kings_same_color_parity"] = int(((wk_r + wk_f) % 2) == ((bk_r + bk_f) % 2))

    features["wk_triangle_mobility"] = int(wk_mob >= 3)
    features["bk_triangle_mobility"] = int(bk_mob >= 3)
    features["wk_has_3cycle"] = int(wk_mob >= 3)
    features["bk_has_3cycle"] = int(bk_mob >= 3)

    lm_w = legal_move_count_for_turn(board, chess.WHITE)
    lm_b = legal_move_count_for_turn(board, chess.BLACK)
    features["move_degree_difference"] = lm_w - lm_b
    features["min_degree_side_to_move"] = legal_move_count_for_turn(board, board.turn)

    features["push_gives_check"] = pawn_push_gives_check_after_push(board, pawn, bk)

    # Optional: keep distance-to-nearest key square (often nonzero)
    features["wk_dist_to_nearest_key"] = min(chebyshev(wk, s) for s in ks) if ks else 0

    return _keep_nonzero(features)