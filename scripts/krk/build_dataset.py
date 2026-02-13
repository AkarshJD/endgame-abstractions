import os
import csv
import chess
import chess.syzygy
from tqdm import tqdm


SYZYGY_PATH = "storage/syzygy/3_4_5"
OUTPUT_PATH = "data/raw/krk/full.csv"


def main():

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading Syzygy tablebases...")
    tablebase = chess.syzygy.open_tablebase(SYZYGY_PATH)

    squares = list(chess.SQUARES)

    rows = []

    print("Enumerating KRK positions...")

    for wk in tqdm(squares, desc="WK square"):
        for wr in squares:
            if wr == wk:
                continue

            for bk in squares:
                if bk == wk or bk == wr:
                    continue

                # Kings cannot be adjacent
                if chess.square_distance(wk, bk) <= 1:
                    continue

                # Create empty board
                board = chess.Board(None)

                board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
                board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

                for turn in [chess.WHITE, chess.BLACK]:

                    board.turn = turn

                    if not board.is_valid():
                        continue

                    try:
                        dtz = tablebase.probe_dtz(board)
                    except:
                        continue

                    if dtz is None:
                        continue

                    rows.append([
                        board.fen(),
                        int(turn == chess.WHITE),
                        dtz
                    ])

    print("Total legal positions:", len(rows))

    print("Saving dataset...")

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "turn", "dtz"])
        writer.writerows(rows)

    print("Done.")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
