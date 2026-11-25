# Experimental Setup

## Oracle: Syzygy Endgame Tablebases

Syzygy tablebases [CITE: Ronald de Man] provide perfect-play ground truth for
all chess positions with up to seven pieces. For each legal position, two
values are available:

- **WDL (Win/Draw/Loss):** outcome under optimal play from both sides.
- **DTZ (Distance to Zeroing):** the minimum number of halfmoves to a pawn move
  or capture under optimal play. Negative values indicate losing positions;
  zero indicates a draw; positive values indicate winning positions.

The 50-move rule states that if 50 moves pass without a pawn move or capture,
the game is drawn by claim. This interacts directly with DTZ: a position is a
"cursed win" if the winning side requires more than 50 halfmoves to zero the
clock — the oracle labels it Win (with optimal play from both sides and no
50-move claim) but in over-the-board play it is a draw.

## Primary Testbed: KBNK

The KBNK endgame (King + Bishop + Knight vs. King) is the primary testbed.
The bishop-knight mating geometry requires driving the opposing king to a
corner of the correct colour — a coordination pattern involving
corner-proximity, colour-parity, and piece-distance features that may create
structurally volatile boundary regions.

The 50-move rule was hypothesised as a second error mechanism: positions
requiring long forced wins might be labelled as draws in practice. This
hypothesis is falsified by full enumeration — see Section 3.3.

**Scale:** All 24,536,088 legal KBNK positions are probed from the oracle for
cursed-position enumeration. For feature extraction and classifier training, 2
million positions are subsampled while preserving the WDL distribution.

**Scale:** We enumerate all legal KBNK positions (24.5 million) via Syzygy
probing and subsample 2 million positions for feature extraction and training,
preserving the WDL distribution.

## Replication Endgames

- **KRK (King + Rook vs. King):** draws arise only via stalemate; the 50-move
  rule plays no role. Expected: DTZ intervention has minimal effect; residual
  errors reflect intrinsic boundary geometry only.

- **KPK (King + Pawn vs. King):** draws arise via pawn blockade and king
  opposition; DTZ is relevant but less structurally dominant than in KBNK.

**KNNK is excluded from this analysis.** Only 736 non-draw positions exist in
12.5 million — accuracy is meaningless as a headline metric, and per-class
fidelity for Win/Loss is undefined at this imbalance. KNNK results would be
reported separately with precision/recall on the minority class only.

## Feature Families

All features are geometric, hand-engineered, and static (no look-ahead):

| Family | Examples |
|---|---|
| Inter-piece distances | Chebyshev and Manhattan king-to-king, knight-to-black-king |
| Corner proximity | Distance to correct/wrong colour corner (KBNK-specific) |
| Edge and mobility | Black king edge distance, geometric mobility count |
| Colour parity | Bishop colour, king square colour, same-colour flags |
| Opposition | King opposition flag, signed rank/file offsets |
| Turn | Side to move (binary) |

**DTZ is excluded by design from the baseline feature set.** The README
explicitly states: "Only WDL files (.rtbw) are needed — this project does not
use DTZ." This is the exclusion the intervention reverses.

## Fidelity Definition

We define per-class fidelity as the fraction of positions in each oracle class
correctly identified by the model:

```
fidelity(C) = P(predict C | oracle = C)
```

This is equivalent to per-class recall. **Accuracy alone is banned as a headline
metric** because it aggregates across classes and masks draw-boundary failures
under majority-class performance.

## Classifier

**Baseline:** `sklearn.tree.DecisionTreeClassifier` with:
- `max_depth=None` — depth is a result, not a constraint
- `min_samples_leaf=50` — forces rules that generalise to at least 50 positions;
  prevents the tree from memorising rare boundary exceptions
- `random_state=42`

**RF comparison (Section 3.4):** `sklearn.ensemble.RandomForestClassifier` with:
- `n_estimators=200`, same `min_samples_leaf=50`, `random_state=42`

Both models use identical 5-fold StratifiedKFold splits and retrain from scratch
per fold (matched-pair CV). No pre-trained model is used in any comparison — both
arms always train and evaluate on the same held-out fold.

DTZ is excluded from all model feature sets. Raw WDL codes (±1/±2) are used
as a label property only — queried post-hoc to classify errors, never fed to any
model.
