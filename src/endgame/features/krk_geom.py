import chess


def chebyshev(a, b):
    return max(
        abs(chess.square_file(a) - chess.square_file(b)),
        abs(chess.square_rank(a) - chess.square_rank(b))
    )


def dist_to_edge(square):
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    return min(rank, 7 - rank, file, 7 - file)


def king_mobility(board, square):
    mobility = 0
    for move in board.generate_legal_moves():
        if move.from_square == square:
            mobility += 1
    return mobility


def min_corner_distance(square):
    corners = [chess.A1, chess.A8, chess.H1, chess.H8]
    return min(chebyshev(square, c) for c in corners)


def compute_krk_features(board):

    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)

    rook = None
    for square in board.pieces(chess.ROOK, chess.WHITE):
        rook = square
        break

    features = {}

    # --- Distances ---
    features["d_wk_bk"] = chebyshev(wk, bk)
    features["d_wr_bk"] = chebyshev(rook, bk)
    features["d_wk_wr"] = chebyshev(wk, rook)

    # --- Edge proximity ---
    features["bk_edge"] = dist_to_edge(bk)
    features["wk_edge"] = dist_to_edge(wk)

    # --- Mobility ---
    features["bk_mobility"] = king_mobility(board, bk)
    features["wk_mobility"] = king_mobility(board, wk)

    # --- Alignment ---
    features["rook_cut_rank"] = int(
        chess.square_rank(rook) == chess.square_rank(bk)
    )
    features["rook_cut_file"] = int(
        chess.square_file(rook) == chess.square_file(bk)
    )

    # --- Confinement geometry ---
    rank_gap = abs(chess.square_rank(rook) - chess.square_rank(bk))
    file_gap = abs(chess.square_file(rook) - chess.square_file(bk))

    features["box_rank_gap"] = rank_gap
    features["box_file_gap"] = file_gap
    features["box_area"] = rank_gap * file_gap

    # --- Edge states ---
    features["bk_on_edge"] = int(features["bk_edge"] == 0)

    is_corner = (
        features["bk_edge"] == 0 and
        chess.square_file(bk) in (0, 7) and
        chess.square_rank(bk) in (0, 7)
    )
    features["bk_in_corner"] = int(is_corner)

    # --- Rook protection ---
    features["rook_protected"] = int(chebyshev(wk, rook) == 1)

    # --- King opposition ---
    same_file = chess.square_file(wk) == chess.square_file(bk)
    same_rank = chess.square_rank(wk) == chess.square_rank(bk)

    opposition = (
        chebyshev(wk, bk) == 2 and
        (same_file or same_rank)
    )

    features["king_opposition"] = int(opposition)

    # --- Corner distance ---
    features["bk_corner_distance"] = min_corner_distance(bk)

    # --- Turn ---
    features["turn"] = int(board.turn == chess.WHITE)

    # --- Rank / file coordinates (signed offsets) ---
    features["wk_rank"] = chess.square_rank(wk)
    features["wk_file"] = chess.square_file(wk)
    features["bk_rank"] = chess.square_rank(bk)
    features["bk_file"] = chess.square_file(bk)
    features["rook_rank"] = chess.square_rank(rook)
    features["rook_file"] = chess.square_file(rook)

    features["wk_bk_rank_diff"] = chess.square_rank(wk) - chess.square_rank(bk)
    features["wk_bk_file_diff"] = chess.square_file(wk) - chess.square_file(bk)

    # --- Parity (critical for triangulation / corresponding squares) ---
    wk_color = (chess.square_file(wk) + chess.square_rank(wk)) % 2
    bk_color = (chess.square_file(bk) + chess.square_rank(bk)) % 2
    rook_color = (chess.square_file(rook) + chess.square_rank(rook)) % 2

    features["wk_square_color"] = wk_color
    features["bk_square_color"] = bk_color
    features["kings_same_color"] = int(wk_color == bk_color)
    features["wk_file_parity"] = chess.square_file(wk) % 2
    features["bk_file_parity"] = chess.square_file(bk) % 2
    features["wk_rank_parity"] = chess.square_rank(wk) % 2
    features["bk_rank_parity"] = chess.square_rank(bk) % 2
    features["rook_file_parity"] = chess.square_file(rook) % 2
    features["rook_rank_parity"] = chess.square_rank(rook) % 2

    # --- Confinement box absolute positions ---
    rr, rf = chess.square_rank(rook), chess.square_file(rook)
    br, bf = chess.square_rank(bk), chess.square_file(bk)
    features["bk_above_rook"] = int(br > rr)
    features["bk_right_of_rook"] = int(bf > rf)

    # --- Manhattan distances ---
    features["m_wk_bk"] = (
        abs(chess.square_file(wk) - chess.square_file(bk)) +
        abs(chess.square_rank(wk) - chess.square_rank(bk))
    )
    features["m_wr_bk"] = (
        abs(chess.square_file(rook) - chess.square_file(bk)) +
        abs(chess.square_rank(rook) - chess.square_rank(bk))
    )

    return features
