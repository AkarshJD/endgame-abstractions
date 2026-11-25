"""
Retrograde solver for K+X vs K endgames on any W x H board.

Syzygy only covers standard 8x8 chess. This module computes exact outcomes by
retrograde analysis for other board sizes and, optionally, other king moves.
On 8x8 it is checked position by position against Syzygy
(scripts/retro/validate_syzygy.py), and elsewhere against an independent
reference solver (scripts/retro/validate_reference.py).

Model
-----
White has a king and one extra piece (Q, R, B, N or P); Black has a lone king.
Squares are (x, y) with 0 <= x < W, 0 <= y < H, flattened as sq = y * W + x.
White pawns move toward y = H - 1 and promote there. Black can never win, so
every position is either a White win or a draw, and the solver only needs to
find the White-win set:

    win_w[s]  - White to move, White wins
    lost_b[s] - Black to move, Black loses

A state s indexes (white king, black king, piece) as s = (wk * N + bk) * N + p,
with N = W * H. The same index is used for both sides to move.

Not modelled: the 50-move rule. For KQK, KRK and KPK on 8x8 it never changes
the result (Syzygy has no cursed wins in these tables), and the validation
script checks that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

KING_STEPS = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0))
ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT_STEPS = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
PAWN_CAPTURES = ((-1, 1), (1, 1))


@dataclass(frozen=True)
class Piece:
    """How the extra White piece moves.

    mode: "slide" (rays until blocked), "leap" (single jumps) or "pawn".
    vectors: ray directions or jump offsets. Ignored for pawns.
    """

    name: str
    mode: str
    vectors: tuple = ()


PIECES = {
    "Q": Piece("Q", "slide", ROOK_DIRS + BISHOP_DIRS),
    "R": Piece("R", "slide", ROOK_DIRS),
    "B": Piece("B", "slide", BISHOP_DIRS),
    "N": Piece("N", "leap", KNIGHT_STEPS),
    "P": Piece("P", "pawn"),
}


@dataclass(frozen=True)
class Rules:
    """The game being solved. Defaults reproduce standard chess on W x H.

    king_steps: how both kings move (and which squares they attack).
        Must be symmetric under negation, e.g. orthogonal-only kings use
        ((1, 0), (-1, 0), (0, 1), (0, -1)).
    pawn_double_step: pawns on y == 1 may advance two squares when both are
        empty (only applied when y == 3 is not the promotion rank).
    promotions: pieces a pawn may promote to. Under-promotion to B or N can
        never beat Q or R in a K+P vs K ending (K+B / K+N vs K are draws), so
        the default leaves them out.
    """

    width: int = 8
    height: int = 8
    king_steps: tuple = KING_STEPS
    pawn_double_step: bool = True
    promotions: tuple = ("Q", "R")

    def __post_init__(self):
        if self.width < 3 or self.height < 3:
            raise ValueError("board must be at least 3x3")
        steps = set(self.king_steps)
        if any((-dx, -dy) not in steps for dx, dy in steps):
            raise ValueError("king_steps must be symmetric under negation")


@dataclass
class Table:
    """Solved table for K + piece vs K under `rules`."""

    rules: Rules
    piece: str
    legal_w: np.ndarray  # White to move, position legal
    legal_b: np.ndarray  # Black to move, position legal
    win_w: np.ndarray  # White to move, White wins
    lost_b: np.ndarray  # Black to move, Black loses
    # Plies to conversion for won positions (-1 otherwise). "Conversion" is
    # mate, or for pawn tables a promotion into a won position, so for KQK/KRK
    # this is exact distance-to-mate and for KPK it is distance to a winning
    # promotion (or mate).
    depth_w: np.ndarray
    depth_b: np.ndarray
    stats: dict = field(default_factory=dict)

    @property
    def n_squares(self) -> int:
        return self.rules.width * self.rules.height

    def index(self, wk: tuple, bk: tuple, p: tuple) -> int:
        w = self.rules.width
        n = self.n_squares
        sq = lambda c: c[1] * w + c[0]  # noqa: E731
        return (sq(wk) * n + sq(bk)) * n + sq(p)

    def wdl(self, wk: tuple, bk: tuple, p: tuple, white_to_move: bool) -> int | None:
        """Syzygy-style WDL from the side to move: 2 win, 0 draw, -2 loss.

        Returns None for illegal positions.
        """
        s = self.index(wk, bk, p)
        if white_to_move:
            return (2 if self.win_w[s] else 0) if self.legal_w[s] else None
        return (-2 if self.lost_b[s] else 0) if self.legal_b[s] else None

    def to_frame(self):
        """All legal positions as a DataFrame (one row per position)."""
        import pandas as pd

        g = _geometry(self.rules.width, self.rules.height)
        frames = []
        for stm, legal, value, depth, sign in (
            ("w", self.legal_w, self.win_w, self.depth_w, 2),
            ("b", self.legal_b, self.lost_b, self.depth_b, -2),
        ):
            idx = np.flatnonzero(legal)
            frames.append(
                pd.DataFrame(
                    {
                        "wk_x": g.wkx[idx], "wk_y": g.wky[idx],
                        "bk_x": g.bkx[idx], "bk_y": g.bky[idx],
                        "p_x": g.px[idx], "p_y": g.py[idx],
                        "stm": stm,
                        "wdl_stm": np.where(value[idx], sign, 0).astype(np.int8),
                        "depth": depth[idx],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            width=self.rules.width,
            height=self.rules.height,
            king_steps=np.array(self.rules.king_steps),
            pawn_double_step=self.rules.pawn_double_step,
            promotions=np.array(self.rules.promotions),
            piece=self.piece,
            legal_w=self.legal_w, legal_b=self.legal_b,
            win_w=self.win_w, lost_b=self.lost_b,
            depth_w=self.depth_w, depth_b=self.depth_b,
        )


# --------------------------------------------------------------------------
# Geometry helpers (vectorised over every state at once)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Geometry:
    width: int
    height: int
    wkx: np.ndarray
    wky: np.ndarray
    bkx: np.ndarray
    bky: np.ndarray
    px: np.ndarray
    py: np.ndarray


@lru_cache(maxsize=8)
def _geometry(width: int, height: int) -> _Geometry:
    n = width * height
    s = np.arange(n**3, dtype=np.int64)
    wk, bk, p = s // (n * n), (s // n) % n, s % n
    to_xy = lambda q: ((q % width).astype(np.int16), (q // width).astype(np.int16))  # noqa: E731
    wkx, wky = to_xy(wk)
    bkx, bky = to_xy(bk)
    px, py = to_xy(p)
    return _Geometry(width, height, wkx, wky, bkx, bky, px, py)


def _inb(x, y, w, h):
    return (x >= 0) & (x < w) & (y >= 0) & (y < h)


def _king_attacks(ax, ay, bx, by, steps) -> np.ndarray:
    """Does a king on a attack square b?"""
    out = np.zeros(ax.shape, dtype=bool)
    dx, dy = bx - ax, by - ay
    for sx, sy in steps:
        out |= (dx == sx) & (dy == sy)
    return out


def _piece_attacks(piece: Piece, px, py, tx, ty, blx, bly, w, h) -> np.ndarray:
    """Does `piece` on p attack square t, with a single possible blocker bl?

    Rays are walked square by square, so any slider direction set works.
    """
    out = np.zeros(px.shape, dtype=bool)
    if piece.mode == "pawn":
        for dx, dy in PAWN_CAPTURES:
            out |= (px + dx == tx) & (py + dy == ty)
        return out
    if piece.mode == "leap":
        for dx, dy in piece.vectors:
            out |= (px + dx == tx) & (py + dy == ty)
        return out
    for dx, dy in piece.vectors:
        alive = np.ones(px.shape, dtype=bool)
        for k in range(1, max(w, h)):
            sx, sy = px + k * dx, py + k * dy
            alive &= _inb(sx, sy, w, h)
            if not alive.any():
                break
            out |= alive & (sx == tx) & (sy == ty)
            alive &= ~((sx == blx) & (sy == bly))
    return out


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

_CACHE: dict = {}


def solve(piece: str | Piece, rules: Rules = Rules(), verbose: bool = False) -> Table:
    """Solve K + piece vs K under `rules`. Results are cached per (piece, rules)."""
    piece = PIECES[piece] if isinstance(piece, str) else piece
    key = (piece, rules)
    if key in _CACHE:
        return _CACHE[key]

    w, h = rules.width, rules.height
    n = w * h
    m = n**3
    g = _geometry(w, h)
    sq = lambda x, y: (y.astype(np.int64) * w + x)  # noqa: E731
    state = lambda wx, wy, bx, by, qx, qy: (sq(wx, wy) * n + sq(bx, by)) * n + sq(qx, qy)  # noqa: E731

    # ---- legality -------------------------------------------------------
    wk_sq, bk_sq, p_sq = sq(g.wkx, g.wky), sq(g.bkx, g.bky), sq(g.px, g.py)
    base = (wk_sq != bk_sq) & (wk_sq != p_sq) & (bk_sq != p_sq)
    base &= ~_king_attacks(g.wkx, g.wky, g.bkx, g.bky, rules.king_steps)
    if piece.mode == "pawn":
        base &= (g.py >= 1) & (g.py <= h - 2)

    bk_in_check = _piece_attacks(piece, g.px, g.py, g.bkx, g.bky, g.wkx, g.wky, w, h)
    legal_w = base & ~bk_in_check  # side not to move may not be in check
    legal_b = base.copy()

    # ---- White moves: edges into Black-to-move states -------------------
    w_src, w_dst = [], []
    promo_win = np.zeros(m, dtype=bool)

    def add_white(mask, dst):
        idx = np.flatnonzero(mask)
        w_src.append(idx.astype(np.int32))
        w_dst.append(dst[idx].astype(np.int32))

    for dx, dy in rules.king_steps:  # White king
        nx, ny = g.wkx + dx, g.wky + dy
        ok = legal_w & _inb(nx, ny, w, h)
        ok &= ~((nx == g.px) & (ny == g.py))
        ok &= ~_king_attacks(nx, ny, g.bkx, g.bky, rules.king_steps)
        add_white(ok, state(nx, ny, g.bkx, g.bky, g.px, g.py))

    if piece.mode == "slide":
        for dx, dy in piece.vectors:
            alive = legal_w.copy()
            for k in range(1, max(w, h)):
                nx, ny = g.px + k * dx, g.py + k * dy
                alive &= _inb(nx, ny, w, h)
                alive &= ~((nx == g.wkx) & (ny == g.wky)) & ~((nx == g.bkx) & (ny == g.bky))
                if not alive.any():
                    break
                add_white(alive, state(g.wkx, g.wky, g.bkx, g.bky, nx, ny))
    elif piece.mode == "leap":
        for dx, dy in piece.vectors:
            nx, ny = g.px + dx, g.py + dy
            ok = legal_w & _inb(nx, ny, w, h)
            ok &= ~((nx == g.wkx) & (ny == g.wky)) & ~((nx == g.bkx) & (ny == g.bky))
            add_white(ok, state(g.wkx, g.wky, g.bkx, g.bky, nx, ny))
    else:  # pawn
        empty = lambda x, y: ~((x == g.wkx) & (y == g.wky)) & ~((x == g.bkx) & (y == g.bky))  # noqa: E731
        ny1 = g.py + 1
        push1 = legal_w & empty(g.px, ny1)
        promoting = push1 & (ny1 == h - 1)
        add_white(push1 & ~promoting, state(g.wkx, g.wky, g.bkx, g.bky, g.px, ny1))
        if rules.pawn_double_step and h - 1 > 3:
            ny2 = g.py + 2
            push2 = push1 & (g.py == 1) & empty(g.px, ny2)
            add_white(push2, state(g.wkx, g.wky, g.bkx, g.bky, g.px, ny2))
        # Promotion: the result is read from the solved K+X vs K table.
        promo_idx = np.flatnonzero(promoting)
        promo_dst = state(g.wkx, g.wky, g.bkx, g.bky, g.px, ny1)[promo_idx]
        for name in rules.promotions:
            sub = solve(name, rules, verbose=verbose)
            promo_win[promo_idx] |= sub.lost_b[promo_dst]

    w_src = np.concatenate(w_src)
    w_dst = np.concatenate(w_dst)

    # ---- Black moves: edges into White-to-move states -------------------
    b_src, b_dst = [], []
    escape = np.zeros(m, dtype=bool)  # Black can capture the piece -> draw
    n_moves_b = np.zeros(m, dtype=np.int16)

    for dx, dy in rules.king_steps:
        nx, ny = g.bkx + dx, g.bky + dy
        ok = legal_b & _inb(nx, ny, w, h)
        ok &= ~_king_attacks(g.wkx, g.wky, nx, ny, rules.king_steps)
        captures = ok & (nx == g.px) & (ny == g.py)  # unprotected, else WK attacks it
        quiet = ok & ~captures
        quiet &= ~_piece_attacks(piece, g.px, g.py, nx, ny, g.wkx, g.wky, w, h)
        escape |= captures
        n_moves_b += captures | quiet
        idx = np.flatnonzero(quiet)
        b_src.append(idx.astype(np.int32))
        b_dst.append(state(g.wkx, g.wky, nx, ny, g.px, g.py)[idx].astype(np.int32))

    b_src = np.concatenate(b_src)
    b_dst = np.concatenate(b_dst)

    mate = legal_b & bk_in_check & (n_moves_b == 0)
    stalemate = legal_b & ~bk_in_check & (n_moves_b == 0)

    # ---- Fixed point (synchronous, so depths are exact) -----------------
    lost_b = mate.copy()
    win_w = np.zeros(m, dtype=bool)
    depth_b = np.where(mate, 0, -1).astype(np.int16)
    depth_w = np.full(m, -1, dtype=np.int16)
    can_lose = legal_b & ~escape & (n_moves_b > 0)

    it = 0
    while True:
        it += 1
        reach_lost = np.bincount(w_src, weights=lost_b[w_dst], minlength=m) > 0
        new_win_w = legal_w & (reach_lost | promo_win)
        not_won = np.bincount(b_src, weights=~new_win_w[b_dst], minlength=m)
        new_lost_b = mate | (can_lose & (not_won == 0))

        depth_w[new_win_w & ~win_w] = 2 * it - 1
        depth_b[new_lost_b & ~lost_b] = 2 * it
        changed = (new_win_w != win_w).any() or (new_lost_b != lost_b).any()
        win_w, lost_b = new_win_w, new_lost_b
        if not changed:
            break
        if verbose:
            print(f"  [{piece.name} {w}x{h}] iter {it}: win_w={win_w.sum():,} lost_b={lost_b.sum():,}")

    stats = {
        "piece": piece.name,
        "board": f"{w}x{h}",
        "legal_w": int(legal_w.sum()),
        "legal_b": int(legal_b.sum()),
        "win_w": int(win_w.sum()),
        "lost_b": int(lost_b.sum()),
        "mates": int(mate.sum()),
        "stalemates": int(stalemate.sum()),
        "max_depth_plies": int(max(depth_w.max(), depth_b.max())),
        "iterations": it,
        "white_edges": int(len(w_src)),
        "black_edges": int(len(b_src)),
    }
    table = Table(rules, piece.name, legal_w, legal_b, win_w, lost_b, depth_w, depth_b, stats)
    _CACHE[key] = table
    return table
