"""Exact solver for K+X vs K endgames on any board size (counterfactual chess)."""

from .solver import PIECES, Piece, Rules, Table, solve

__all__ = ["PIECES", "Piece", "Rules", "Table", "solve"]
