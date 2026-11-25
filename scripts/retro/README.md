# Exact K+X vs K on other boards

`src/endgame/retro/solver.py` solves K+Q, K+R, K+B, K+N and K+P vs K by
retrograde analysis on a W x H board, optionally with different king moves
(for example kings that only move orthogonally). Syzygy covers only the
standard 8x8 game.

## Trust

On the standard 8x8 board the solver agrees with Syzygy on every legal
position, and its legal set is identical to python-chess's:

| Endgame | Positions checked | WDL agreement | Legality disagreements |
|---|---|---|---|
| KQK | 368,452 | 100% | 0 |
| KRK | 399,112 | 100% | 0 |
| KPK | 331,352 | 100% | 0 |

Rerun with `python scripts/retro/validate_syzygy.py` (about 15 s with 7 workers).
`tests/test_retro.py` adds known positions, board-symmetry checks and a Syzygy
spot check.

On other boards and with other king moves the solver is checked against
`src/endgame/retro/reference.py`, a second solver written independently: plain
Python, one position at a time, every move played on a board and kept only if
the resulting position is legal. It shares no code with the solver (a test
enforces that). Legality, WDL and depth agree on every position:

| Kings | Boards | Endgames | Positions | Mismatches |
|---|---|---|---|---|
| standard | 5x5, 6x6, 7x7, 8x8, 7x10, 10x7 | KQK, KRK, KPK | 4,699,366 | 0 |
| orthogonal | 5x5, 6x6, 7x7, 8x8, 7x10, 10x7 | KQK, KRK, KPK | 4,958,964 | 0 |

Rerun with `python scripts/retro/validate_reference.py` (about 2 min with 4
workers); `tests/test_retro_reference.py` runs KQK, KRK, KBK, KNK and KPK on
5x5, 4x6 and 6x4 in about 10 s.

As a one-off check (not in the test suite), single bugs were planted in a copy
of the solver: pawn capture direction, kings allowed adjacent, sliders passing
through kings, captures not counted as escapes, no double step, queen-only
promotion, stalemate scored as mate, an illegal check position allowed, a
depth off by one. The comparison failed for every one.

What this does not cover: both solvers implement the same written rules, so a
mistake in the rules themselves (rather than in the code) would not show up.
The rules are spelled out at the top of `reference.py`.

## Use

```bash
python scripts/retro/solve_boards.py                       # all sizes, both king variants
python scripts/retro/solve_boards.py --sizes 6x6 10x10 --pieces P --kings standard
```

Tables go to `data/processed/retro/<kings>/<W>x<H>/K<X>K.npz` (gitignored),
with a summary in `data/processed/retro/summary.csv`. The full default set
(5x5 to 10x10, 7x10, 10x7, standard and orthogonal kings) takes under 30 s.

```python
from endgame.retro import Rules, solve
from endgame.retro.solver import ROOK_DIRS

t = solve("P", Rules(width=10, height=10))           # KPK on 10x10
df = t.to_frame()                                    # one row per legal position
alt = solve("P", Rules(king_steps=ROOK_DIRS))        # kings move orthogonally only
```

`wdl_stm` follows Syzygy: 2 win, 0 draw, -2 loss for the side to move. `depth`
is plies to conversion: exact distance to mate for KQK/KRK, distance to a
winning promotion (or mate) for KPK.

## Conventions and limits

- Squares are (x, y); White pawns move toward y = H - 1, double-step from
  y = 1 when y = 3 is not the promotion rank, and promote to Q or R.
- No 50-move rule. For KQK, KRK and KPK on 8x8 this never changes a result
  (Syzygy reports zero cursed positions), but it may on larger boards.
- Boards up to about 10x10 fit comfortably in 16 GB; memory grows as (W*H)^3.
