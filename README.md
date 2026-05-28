# Endgame Abstractions

Decision tree classifiers trained on geometric features to predict chess endgame
WDL (Win/Draw/Loss) from Syzygy tablebases. No search — pure geometric pattern
recognition.

The central question: how much of chess endgame knowledge is recoverable from
geometry alone, without search?

## Results

| Endgame | Positions | Depth | Leaves | Accuracy | Key finding |
|---|---|---|---|---|---|
| KRK | ~900K | 5 | 8 | 99.983% | 8 rules describe the entire endgame |
| KBBK | 11.9M | 16 | 72 | 99.956% | Root split: `bishops_same_color` (insufficient material) |
| KPK | ~331K | 16 | 548 | ~99.9% | Promotion explodes complexity 68× vs KRK |
| KBNK | 24.5M | 24 | 770 | 99.540% | 50-move rule creates boundary geometry can't cross |
| KNNK | 12.5M | 8 | 15 | 99.994% | 736 non-draws in 12.5M — theoretical draw recovered by ML |

**Universal error pattern:** every misclassification in every endgame involves the
DRAW boundary. Zero WIN↔LOSS errors. Geometry perfectly separates winning from
losing positions; the draw boundary is what it cannot precisely specify.

## Approach

1. Enumerate all legal positions via Syzygy WDL probing
2. Compute geometric features per position (distances, parity, coverage)
3. Train `DecisionTreeClassifier(min_samples_leaf=50, max_depth=None)`
4. Validate per-class accuracy; analyze misclassified positions

Chess knowledge is not manually encoded. All abstractions are learned from data.

**Feature design highlights:**
- `turn` dominates every endgame except KBBK (81.8% importance in KRK)
- KBBK: `bishops_same_color` at 50.2% — material fact outweighs tempo
- KBNK: `bk_in_wrong_corner` / `bk_near_wrong_corner` are zero importance — the tree uses raw distances, not named concepts
- KPK: `e2_bk_p` (rule of the square) appears in top-5 features — a named chess heuristic recovered from data
- KNNK: `nn_overlap` (squares both knights attack simultaneously) found without being told about coordination

## Repository Structure

```
scripts/{endgame}/
    build_dataset.py    — enumerate positions, probe Syzygy WDL, save CSV
    build_features.py   — extract geometric features, save features CSV
    train_classifier.py — train DecisionTree, save model + tree.txt + metrics
    validate.py         — accuracy per class, misclassified FEN positions
    run_pipeline.sh     — runs all 4 steps in order

scripts/analysis/
    feature_importance.py  — cross-endgame feature ranking with chess-concept labels

src/endgame/features/
    krk_geom.py   kpk_geom.py   kbnk_geom.py   knnk_geom.py   kbbk_geom.py

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

Only WDL files (`.rtbw`) are needed — this project does not use DTZ.

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
  prevents the tree from fitting the rare exceptions we study separately
- `max_depth=None` — depth is a result (finding), not a constraint
- Syzygy WDL only (no DTZ) — halves probe time, sufficient for classification
- 2M position sample for 4-piece endgames — full 24.5M would OOM at 16GB
- Symmetric pieces enumerated with `p1 < p2` to avoid duplicates

## Acknowledgments

This project uses [Syzygy endgame tablebases](https://github.com/syzygy1/tb) by
Ronald de Man, distributed by Lichess.

## Author

Akarsh J D — Active Research
