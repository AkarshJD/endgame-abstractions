"""
Independent reference solver for K + X vs K, for checking solver.py.

Syzygy can only check solver.py on the standard 8x8 board with chess kings.
This module checks it everywhere else (other board sizes, other king moves)
by solving the same games a second, unrelated way. It shares no code with
solver.py on purpose, and tests/test_retro_reference.py fails if it ever
imports it.

    solver.py     numpy, every state at once; whether a move is legal is a
                  hand-derived formula (e.g. "the black king may step to t if
                  t is not next to the white king and the piece, with the
                  white king as its only blocker, does not attack t")
    reference.py  plain Python, one position at a time; every move is played
                  on a board and kept only if the resulting position is legal
                  (the textbook rule: you may not leave your king attacked)

Rules, written from the chess rules rather than from solver.py:
  * Board W x H, squares (x, y). White pawns move toward y = H - 1.
  * Both kings move, and attack, by `king_steps`.
  * A position is legal when the pieces stand on distinct squares, a pawn is
    not on the first or last rank, and the side NOT to move is not attacked.
    A move is legal when the position it produces is legal.
  * A pawn advances one square into an empty square, or two from y == 1 when
    both squares are empty and it does not land on the last rank. Reaching the
    last rank it promotes to one of `promotions`.
  * Black has only the king. Capturing White's piece leaves K vs K, a draw.
  * No legal move: checkmate if attacked, otherwise stalemate (a draw).
  * No 50-move or repetition rule; a game that never ends is a draw.

Depth is plies to conversion (mate, or for pawn tables a promotion into a
position lost for Black). It is built one ply layer at a time, which gives
exactly:
  White to move, won:  1 + min depth of the moves that reach a lost position
  Black to move, lost: 1 + max depth over all moves (0 when mated)

Pure Python, so keep boards small: a 5x5 table takes about a second, 6x6
a few seconds.
"""

from __future__ import annotations

from functools import lru_cache

ORTHOGONAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))

# How White's extra piece moves and attacks, apart from the pawn.
RAYS = {"Q": ORTHOGONAL + DIAGONAL, "R": ORTHOGONAL, "B": DIAGONAL}
JUMPS = {"N": KNIGHT}

# Children that are not positions of the table being solved.
CONVERTED = "converted"  # promotion into a position lost for Black
DRAWN = "drawn"  # K vs K after a capture, or a promotion that does not win


class _Game:
    def __init__(self, width, height, king_steps, double_step, promotions):
        self.w, self.h = width, height
        self.king_steps = frozenset(king_steps)
        self.double_step = double_step
        self.promotions = promotions

    def on_board(self, sq) -> bool:
        return 0 <= sq[0] < self.w and 0 <= sq[1] < self.h

    def attacks(self, kind, frm, target, occupied) -> bool:
        """Does a piece of `kind` on `frm` attack `target`?"""
        dx, dy = target[0] - frm[0], target[1] - frm[1]
        if kind == "K":
            return (dx, dy) in self.king_steps
        if kind == "P":
            return dy == 1 and abs(dx) == 1
        if kind in JUMPS:
            return (dx, dy) in JUMPS[kind]
        for sx, sy in RAYS[kind]:
            x, y = frm[0] + sx, frm[1] + sy
            while self.on_board((x, y)):
                if (x, y) == target:
                    return True
                if (x, y) in occupied:
                    break
                x, y = x + sx, y + sy
        return False

    def attacked_by_white(self, pos, target) -> bool:
        wk, bk, kind, p = pos
        occupied = {wk, bk} if p is None else {wk, bk, p}
        if self.attacks("K", wk, target, occupied):
            return True
        return kind is not None and self.attacks(kind, p, target, occupied)

    def legal(self, pos, white_to_move: bool) -> bool:
        wk, bk, kind, p = pos
        squares = [wk, bk] if p is None else [wk, bk, p]
        if len(set(squares)) != len(squares) or not all(map(self.on_board, squares)):
            return False
        if kind == "P" and p[1] in (0, self.h - 1):
            return False
        if white_to_move:
            return not self.attacked_by_white(pos, bk)  # Black, not to move, is not in check
        return not self.attacks("K", bk, wk, None)  # White, not to move, is not in check

    # ---- moves: play every candidate, keep those that give a legal position

    def white_moves(self, pos):
        """Positions after each legal White move (Black to move)."""
        wk, bk, kind, p = pos
        out = []
        for sx, sy in self.king_steps:
            t = (wk[0] + sx, wk[1] + sy)
            if t not in (bk, p):
                out.append((t, bk, kind, p))
        if kind == "P":
            one = (p[0], p[1] + 1)
            if one not in (wk, bk):
                if one[1] == self.h - 1:
                    out += [(wk, bk, promo, one) for promo in self.promotions]
                else:
                    out.append((wk, bk, "P", one))
                    two = (p[0], p[1] + 2)
                    if (self.double_step and p[1] == 1 and two not in (wk, bk)
                            and self.on_board(two) and two[1] != self.h - 1):
                        out.append((wk, bk, "P", two))
        elif kind in JUMPS:
            for sx, sy in JUMPS[kind]:
                t = (p[0] + sx, p[1] + sy)
                if t not in (wk, bk):
                    out.append((wk, bk, kind, t))
        else:
            for sx, sy in RAYS[kind]:
                t = (p[0] + sx, p[1] + sy)
                while self.on_board(t) and t not in (wk, bk):
                    out.append((wk, bk, kind, t))
                    t = (t[0] + sx, t[1] + sy)
        return [c for c in out if self.legal(c, white_to_move=False)]

    def black_moves(self, pos):
        """Positions after each legal Black move (White to move)."""
        wk, bk, kind, p = pos
        out = []
        for sx, sy in self.king_steps:
            t = (bk[0] + sx, bk[1] + sy)
            if t == p:
                out.append((wk, t, None, None))  # captures White's piece
            elif t != wk:
                out.append((wk, t, kind, p))
        return [c for c in out if self.legal(c, white_to_move=True)]


@lru_cache(maxsize=None)
def solve_reference(
    piece: str,
    width: int,
    height: int,
    king_steps: tuple,
    double_step: bool = True,
    promotions: tuple = ("Q", "R"),
) -> dict:
    """Solve K + piece vs K. Returns {(wk, bk, p, white_to_move): (wdl, depth)}.

    Squares are (x, y) tuples. wdl is from the side to move (2 win, 0 draw,
    -2 loss); depth is plies to conversion, -1 for draws. Only legal positions
    are included.
    """
    game = _Game(width, height, tuple(king_steps), double_step, tuple(promotions))
    squares = [(x, y) for y in range(height) for x in range(width)]

    promoted = {}
    if piece == "P":
        for name in promotions:
            promoted[name] = solve_reference(name, width, height, king_steps, double_step, promotions)

    def label(child):
        """Children outside this table become CONVERTED or DRAWN."""
        kind = child[2]
        if kind is None:
            return DRAWN
        if kind != piece:  # a promotion
            wk, bk, _, p = child
            return CONVERTED if promoted[kind][(wk, bk, p, False)][0] == -2 else DRAWN
        return child

    white, black = {}, {}  # position -> list of children
    for wk in squares:
        for bk in squares:
            for p in squares:
                pos = (wk, bk, piece, p)
                if game.legal(pos, white_to_move=True):
                    white[pos] = [label(c) for c in game.white_moves(pos)]
                if game.legal(pos, white_to_move=False):
                    black[pos] = [label(c) for c in game.black_moves(pos)]

    won = {}  # White to move, White wins: position -> depth
    lost = {pos: 0 for pos, kids in black.items() if not kids and game.attacked_by_white(pos, pos[1])}

    ply = 0
    while True:
        ply += 1
        new_won = [pos for pos, kids in white.items() if pos not in won
                   and any(c == CONVERTED or c in lost for c in kids)]
        won.update((pos, ply) for pos in new_won)
        ply += 1
        new_lost = [pos for pos, kids in black.items() if pos not in lost
                    and kids and all(c in won for c in kids)]
        lost.update((pos, ply) for pos in new_lost)
        if not new_won and not new_lost:
            break

    out = {}
    for pos in white:
        out[(pos[0], pos[1], pos[3], True)] = (2, won[pos]) if pos in won else (0, -1)
    for pos in black:
        out[(pos[0], pos[1], pos[3], False)] = (-2, lost[pos]) if pos in lost else (0, -1)
    return out
