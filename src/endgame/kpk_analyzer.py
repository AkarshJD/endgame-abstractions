"""
KPK Endgame Analyzer

Purpose:
- Analyze King + Pawn vs King positions
- Combine SyZYGY truth with human-style reasoning
"""

import chess
import chess.syzygy
from pathlib import Path

from configs.paths import SYZYGY_PATH


class KPKAnalyzer:
    def __init__(self, tb_path: Path = SYZYGY_PATH):
        self.tb_path = tb_path
        self.tablebase = chess.syzygy.open_tablebase(str(tb_path))

    def is_kpk(self, board: chess.Board) -> bool:
        """
        Check if position is exactly K+P vs K
        """
        pieces = board.piece_map().values()

        kings = sum(1 for p in pieces if p.piece_type == chess.KING)
        pawns = sum(1 for p in pieces if p.piece_type == chess.PAWN)

        return kings == 2 and pawns == 1 and len(pieces) == 3

    def probe(self, board: chess.Board):
        """
        Query Syzygy for WDL and DTZ
        """
        wdl = self.tablebase.probe_wdl(board)
        dtz = self.tablebase.probe_dtz(board)

        return wdl, dtz

    def analyze(self, fen: str) -> dict:
        """
        Main entry point
        """
        board = chess.Board(fen)

        if not self.is_kpk(board):
            raise ValueError("Not a KPK position")

        wdl, dtz = self.probe(board)

        result = self._interpret_wdl(wdl)

        explanation = self._explain_position(board, wdl, dtz)

        return {
            "fen": fen,
            "result": result,
            "wdl": wdl,
            "dtz": dtz,
            "explanation": explanation,
        }

    def _interpret_wdl(self, wdl: int) -> str:
        """
        Convert WDL to human label
        """
        if wdl > 0:
            return "Win"
        elif wdl < 0:
            return "Loss"
        else:
            return "Draw"

    def _explain_position(
        self,
        board: chess.Board,
        wdl: int,
        dtz: int
    ) -> str:
        """
        Placeholder for human-style reasoning
        """
        lines = []

        if wdl > 0:
            lines.append("Side to move can force promotion.")
        elif wdl == 0:
            lines.append("Position is theoretically drawn.")
        else:
            lines.append("Side to move is losing.")

        lines.append(f"Distance to zeroing move: {dtz}")

        return "\n".join(lines)


if __name__ == "__main__":
    analyzer = KPKAnalyzer()

    test_fen = "8/8/4k3/8/4P3/4K3/8/8 w - - 0 1"

    board = chess.Board(test_fen)

    result = analyzer.analyze(test_fen)

    print("=" * 40)
    print("Position (FEN):")
    print(test_fen)
    print()

    print("Board:")
    print(board.unicode())
    print()

    print("Result:", result["result"])
    print("DTZ:", result["dtz"])
    print()

    print("Explanation:")
    print(result["explanation"])
    print("=" * 40)

