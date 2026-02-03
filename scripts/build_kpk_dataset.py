"""
Build full KPK state space with perfect labels
using Syzygy tablebases.

Enumerates all geometrically legal KPK positions.
"""
import csv
import chess
import chess.syzygy
from pathlib import Path

from configs.paths import SYZYGY_PATH


OUT_FILE = Path("data/processed/kpk/full.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def is_kpk(board: chess.Board) -> bool:
    pieces = board.piece_map().values()

    counts = {
        chess.KING: 0,
        chess.PAWN: 0,
        chess.QUEEN: 0,
        chess.ROOK: 0,
        chess.BISHOP: 0,
        chess.KNIGHT: 0,
    }

    for p in pieces:
        counts[p.piece_type] += 1

    return (
        counts[chess.KING] == 2
        and counts[chess.PAWN] == 1
        and sum(counts.values()) == 3
    )


def generate_kpk_positions():
    board = chess.Board(None)

    squares = list(chess.SQUARES)

    for wk in squares:
        for bk in squares:
            if wk == bk:
                continue

            if chess.square_distance(wk, bk) <= 1:
                continue

            for pawn in squares:
                if pawn in (wk, bk):
                    continue

                for color in [chess.WHITE, chess.BLACK]:

                    board.clear()

                    board.turn = color

                    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                    board.set_piece_at(pawn, chess.Piece(chess.PAWN, chess.WHITE))

                    if not board.is_valid():
                        continue

                    yield board.copy()


def main():

    tb = chess.syzygy.open_tablebase(SYZYGY_PATH)

    print("Generating KPK dataset...")

    rows = []

    for i, board in enumerate(generate_kpk_positions()):

        try:
            wdl = tb.probe_wdl(board)
            dtz = tb.probe_dtz(board)
        except:
            continue

        rows.append([
            board.fen(),
            int(board.turn),
            wdl,
            dtz
        ])

        if i % 10000 == 0:
            print("Processed:", i)

    with open(OUT_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "fen",
            "turn",
            "wdl",
            "dtz"
        ])

        writer.writerows(rows)

    tb.close()

    print("Saved:", OUT_FILE)
    print("Rows:", len(rows))


if __name__ == "__main__":
    main()
