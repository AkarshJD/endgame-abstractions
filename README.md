# Endgame Abstractions

Decision tree classifiers trained on hand-engineered geometric features to predict
chess endgame WDL (Win/Draw/Loss), using Syzygy tablebases as a perfect oracle for
ground truth.

Two things are deliberately separated. The **classifier** performs no game-tree
search and encodes no chess rules — it is a decision tree over numeric features. The
**feature basis**, by contrast, is hand-engineered from board geometry. The purpose
of that split is not to play chess well; it is to locate precisely where a static,
search-free geometric representation can and cannot reproduce tablebase-perfect
classification.

## What this project is trying to show

Precise version of the question:

> Given a perfect oracle, where does a search-free geometric representation succeed
> and fail at separating Win / Draw / Loss?

The proposition the experiments are set up to test — and, where possible, to
falsify — is:

> In each endgame studied, a decision tree over geometric features separates
> decisive positions from draws almost perfectly, and the remaining errors sit on
> the draw boundary.

Win vs Loss is not part of the claim. Syzygy WDL is scored from the side to move,
and the lone king can never win, so in all five datasets every win has White to
move and every loss has Black to move. Any tree that splits on side to move
cannot confuse the two.

This is a claim about the geometry of the problem, not about the strength of the
model. Two properties make it worth stating rather than asserting:

- **Oracle-grounded.** Labels come from Syzygy WDL probing, which is exact. There is
  no label noise, so every misclassification is a genuine boundary that geometry
  failed to specify — not a possibly-mislabeled position.
- **Falsification-tested.** The pipeline is built to catch its own failures. An
  earlier feature set included DTZ, which is derived from the same oracle as the
  labels, and produced a tautological near-perfect result; it was removed (see
  *Leakage and oracle hygiene*).

### What this project does *not* claim

- It does **not** claim the *mechanism* of the draw boundary is understood. One
  candidate mechanism (the 50-move rule) was tested and ruled out for KBNK — the
  capture threshold is reachable well within the 100-ply limit, so it cannot be what
  geometry is missing. The boundary remains unexplained in general. This is the main
  open question, not a settled result.
- It does **not** claim chess concepts *emerge* from raw board state. Where a named
  concept (e.g. the rule of the square) shows up, it shows up because a feature
  encoding that concept was supplied. Whether comparable structure would arise from
  primitive features alone is untested (see *Open questions*).

## Results

| Endgame | Positions | Depth | Leaves | Test accuracy | Side to move only |
|---|---|---|---|---|---|
| KRK  | 399K   | 5  | 8   | 99.98% | 94.43% |
| KBBK | 11.9M† | 16 | 72  | 99.95% | 55.26% |
| KPK  | 331K   | 16 | 548 | 98.25% | 67.17% |
| KBNK | 24.5M† | 24 | 770 | 99.51% | 89.70% |
| KNNK | 12.6M† | 8  | 15  | 99.99% | 99.99% |

- *Test accuracy* is on the held-out split from `train_classifier.py`. Earlier
  versions of this table used `validate.py`, which scores every position,
  including the ones the tree was trained on; the earlier KPK figure (~99.9%)
  also came from an older tree grown with no leaf-size limit.
- *Side to move only* is the accuracy of predicting the most common result for
  each side to move, ignoring the board. The gap between the two columns is what
  the geometric features add.
- † Trained and evaluated on a 2M-position sample.

Accuracy is dominated by base rates and should be read with the per-class recall
and confusion matrix from `validate.py`. There are zero Win↔Loss errors in all
five endgames, but as explained above this follows from the labels, not from
geometry.

Per-endgame notes:

- **KRK** — 8 leaves reach 99.98% on held-out positions. Every White-to-move
  position in the dataset is a win, so the tree's real job is telling
  Black-to-move losses from draws.
- **KBBK** — root split is `bishops_same_color`, a material fact (same-colored
  bishops cannot force mate) outweighing tempo. Reassuring that the tree's primary
  split is structurally correct.
- **KPK** — 548 leaves, about 68× more than KRK. An engineered
  `e2_bk_p` feature (rule of the square) ranks in the top 5. The tree *uses* it; this
  is **not** evidence the rule of the square emerges from board geometry, only that a
  feature already encoding it is discriminative.
- **KBNK** — the named `bk_in_wrong_corner` / `bk_near_wrong_corner` features carry
  zero importance; the tree relies on raw distances instead. A case where a
  human-named concept was *not* what the tree found useful, which is worth reporting
  honestly rather than hiding.
- **KNNK** — KNN vs K is a theoretical draw: 736 of the 12.6M positions are not
  draws. The tree predicts a draw for every held-out position (0 of 36 non-draws
  correct), so its accuracy is just the base rate. There is no KNNK result yet.

## Approach

1. Enumerate legal positions via Syzygy WDL probing.
2. Compute geometric features per position (distances, parity, coverage).
3. Train `DecisionTreeClassifier(min_samples_leaf=50, max_depth=None)`.
4. Report per-class recall and the confusion matrix; collect misclassified FENs.

No chess *rules* are encoded in the classifier and no search is performed. Chess
knowledge does enter the system — through feature design. The geometric vocabulary
(distances, parity, coverage, and a few named-concept features) is hand-built. The
split thresholds *within* that vocabulary are learned from data; the vocabulary
itself is not. Claims of the form "learned from data" apply to the thresholds, not to
the representation.

## Leakage and oracle hygiene

Syzygy DTZ (distance-to-zero) is a function of the same oracle that produces the
labels. An earlier feature set that included DTZ produced near-perfect numbers that
were tautological — the feature was quietly carrying the answer. DTZ was removed.
Only WDL (`.rtbw`) files are used, which is why the setup below downloads nothing
else. General rule adopted here: any feature that is a function of the oracle is
treated as leakage and excluded.

## Open questions

These are stated as open, not as findings:

1. **Draw-boundary mechanism.** What governs the width and shape of the draw
   boundary that the geometric features cannot place exactly? The 50-move rule was tested
   and rejected for KBNK. Candidate correlates still to test: tempo/opposition
   parity, distance to zugzwang, and reachability of a decisive capture or promotion
   within the relevant horizon.
2. **Concept emergence vs. concept supply.** Do named heuristics (rule of the square,
   knight coordination) reappear in the split structure when *only* primitive
   features — raw coordinates, parity, side-to-move — are provided? If yes,
   "recovered from data" becomes an earned claim. If no, the concept is doing work
   primitives can't reconstruct. Either outcome is informative.
3. **Portability.** Is "all error is draw-adjacent" a property of these specific small
   endgames, or a general property of geometric WDL separability? This needs more and
   larger endgames before any general claim is made.

## Known issues

- The current KPK feature file stores `turn` as 1 for White to move and leaves it
  empty (NaN) for Black to move. The other endgames use 1 and 0.

## Repository Structure

```
scripts/{endgame}/
    build_dataset.py    — enumerate positions, probe Syzygy WDL, save CSV
    build_features.py   — extract geometric features, save features CSV
    train_classifier.py — train DecisionTree, save model + tree.txt + metrics
    validate.py         — per-class recall, confusion matrix, misclassified FENs
    run_pipeline.sh     — runs all 4 steps in order

scripts/analysis/
    feature_importance.py  — cross-endgame feature ranking with chess-concept labels

src/endgame/features/
    krk_geom.py   kpk_geom.py   kbnk_geom.py   knnk_geom.py   kbbk_geom.py

src/endgame/retro/      — exact K+X vs K solver for other board sizes and king moves
scripts/retro/          — solve tables, check them against Syzygy and a second solver
                          (see scripts/retro/README.md)

data/raw/{endgame}/full.csv            — gitignored, regenerate with build_dataset.py
data/processed/{endgame}/features.csv — gitignored, regenerate with build_features.py
logs/{endgame}_classifier/            — gitignored, contains model.joblib + tree.txt
storage/syzygy/3_4_5/                 — gitignored, local Syzygy .rtbw files
```

## Installation

```bash
git clone https://github.com/AkarshJD/endgame-abstractions.git
cd endgame-abstractions

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Syzygy Setup

```bash
mkdir -p storage/syzygy/3_4_5
```

Download 3-4-5 WDL files from [tablebase.lichess.ovh](https://tablebase.lichess.ovh/tables/standard/)
and place all `.rtbw` files into `storage/syzygy/3_4_5/`.

Only WDL files (`.rtbw`) are needed. DTZ is deliberately excluded — see
*Leakage and oracle hygiene*.

## Running a Pipeline

```bash
source venv/bin/activate
bash scripts/{endgame}/run_pipeline.sh
```

Or step by step:

```bash
python scripts/{endgame}/build_dataset.py    # ~75 min for 4-piece (7 workers)
python scripts/{endgame}/build_features.py   # ~5 min (parallelized, 2M sample)
python scripts/{endgame}/train_classifier.py
python scripts/{endgame}/validate.py
```

Replace `{endgame}` with `krk`, `kpk`, `kbnk`, `knnk`, or `kbbk`.

## Analysis

```bash
python scripts/analysis/feature_importance.py   # feature rankings across all endgames
python scripts/kpk/compression_analysis.py      # compression ratio vs raw tablebase
```

## Key Design Decisions

- `min_samples_leaf=50` — forces rules that generalize to at least 50 positions;
  prevents the tree from fitting rare exceptions, which are studied separately.
- `max_depth=None` — depth is a reported result, not an imposed constraint.
- **WDL only, no DTZ** — primarily to avoid oracle leakage (DTZ is oracle-derived);
  halving probe time is a secondary benefit.
- 2M-position sample for 4-piece endgames — full 24.5M would OOM at 16GB.
- Symmetric pieces enumerated with `p1 < p2` to avoid duplicates.

## Acknowledgments

This project uses [Syzygy endgame tablebases](https://github.com/syzygy1/tb) by
Ronald de Man, distributed by Lichess.

## Author

Akarsh J D — Active Research
