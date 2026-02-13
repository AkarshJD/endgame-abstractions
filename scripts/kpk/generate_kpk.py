import chess
import random


def random_square(exclude=None):
    """
    Pick a random square, optionally excluding some.
    """
    squares = list(chess.SQUARES)

    if exclude:
        squares = [s for s in squares if s not in exclude]

    return random.choice(squares)


def kings_adjacent(white_king, black_king):
    """
    Check if kings touch (illegal).
    """
    return chess.square_distance(white_king, black_king) <= 1


def valid_pawn_square(square, color):
    """
    Pawns cannot be on first/last rank.
    """
    rank = chess.square_rank(square)

    if color == chess.WHITE:
        return rank not in (0, 7)
    else:
        return rank not in (0, 7)


def generate_kpk_position():
    """
    Generate a legal random KPK position.
    Returns a chess.Board object.
    """

    board = chess.Board(None)  # Empty board

    while True:

        # Pick king squares
        wk = random_square()
        bk = random_square(exclude=[wk])

        if kings_adjacent(wk, bk):
            continue

        # Pick pawn square (white pawn)
        pawn_sq = random_square(exclude=[wk, bk])

        if not valid_pawn_square(pawn_sq, chess.WHITE):
            continue

        # Place pieces
        board.clear()

        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.WHITE))

        # Side to move: random
        board.turn = random.choice([chess.WHITE, chess.BLACK])

        # Check legality
        if not board.is_valid():
            continue

        # No side already checkmated/stalemated
        if board.is_checkmate() or board.is_stalemate():
            continue

        return board


def main(n=1000):

    print("Generating KPK positions...")

    for i in range(n):
        b = generate_kpk_position()
        print(b.fen())


if __name__ == "__main__":
    main(1000)
