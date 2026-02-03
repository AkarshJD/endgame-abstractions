import chess


def chebyshev(a, b):
    return max(
        abs(chess.square_file(a) - chess.square_file(b)),
        abs(chess.square_rank(a) - chess.square_rank(b))
    )


def extract_kpk_features(board: chess.Board):

    pieces = board.piece_map()

    wk = None
    bk = None
    pawn = None

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

    # Distances
    features["d_wk_bk"] = chebyshev(wk, bk)
    features["d_wk_p"] = chebyshev(wk, pawn)
    features["d_bk_p"] = chebyshev(bk, pawn)

    # Pawn progress
    features["pawn_rank"] = pawn_rank
    features["d_promo"] = promotion_rank - pawn_rank

    # Alignment
    features["wk_bk_file"] = int(
        chess.square_file(wk) == chess.square_file(bk)
    )

    features["wk_bk_rank"] = int(
        chess.square_rank(wk) == chess.square_rank(bk)
    )

    features["rank_gap"] = abs(
        chess.square_rank(wk) - chess.square_rank(bk)
    )

    features["file_gap"] = abs(
        chess.square_file(wk) - chess.square_file(bk)
    )

    # Tempo
    features["turn"] = int(board.turn)

    return features
