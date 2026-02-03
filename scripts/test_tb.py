import chess
import chess.syzygy
import configs
from configs.paths import SYZYGY_PATH


def main():
    board = chess.Board("8/8/8/8/8/8/4K3/4k3 w - - 0 1")

    with chess.syzygy.open_tablebase(str(SYZYGY_PATH)) as tb:
        wdl = tb.probe_wdl(board)
        dtz = tb.probe_dtz(board)

    print("Position:")
    print(board)
    print("WDL:", wdl)
    print("DTZ:", dtz)


if __name__ == "__main__":
    main()
