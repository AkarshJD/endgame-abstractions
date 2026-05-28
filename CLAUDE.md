# endgame-abstractions

Decision tree classifiers trained on geometric features to predict chess endgame WDL
(Win/Draw/Loss) from Syzygy tablebases. No search — pure geometric pattern recognition.

## Results so far

| Endgame | Positions | Depth | Leaves | Accuracy | Key finding |
|---|---|---|---|---|---|
| KRK | ~900K | 5 | 8 | 99.983% | 8 rules describe entire endgame |
| KBBK | 11.9M | 16 | 72 | 99.956% | Root = `bishops_same_color` (insufficient material) |
| KPK | ~331K | 16 | 548 | ~99.9% | Promotion explodes complexity 68× vs KRK |
| KBNK | 24.5M | 24 | 770 | 99.540% | 50-move rule creates boundary geometry can't cross |
| KNNK | 12.5M | 8 | 15 | 99.994% | 736 non-draws in 12.5M — theoretical draw proven by ML |

All misclassifications in every endgame involve the DRAW class. Zero WIN↔LOSS errors.

## Stack

- `python-chess` — FEN parsing, legal move generation, Syzygy probing
- `scikit-learn` DecisionTreeClassifier — WDL classifier
- `pandas` / `joblib` — feature CSVs and model serialization
- `multiprocessing.Pool` — parallel Syzygy enumeration (7 workers, ~14× speedup)

## Repo layout

```
scripts/{endgame}/
    build_dataset.py   — enumerate positions, probe Syzygy WDL, save CSV
    build_features.py  — extract geometric features, save features CSV
    train_classifier.py — train DecisionTree, save model + tree.txt + metrics
    validate.py        — accuracy per class, misclassified FEN positions
    run_pipeline.sh    — runs all 4 steps in order

src/endgame/features/
    krk_geom.py        — KRK geometric features
    kpk_geom.py        — KPK geometric features
    kbnk_geom.py       — KBNK geometric features (bishop color, correct/wrong corner)
    knnk_geom.py       — KNNK geometric features (knight coverage, BK mobility)
    kbbk_geom.py       — KBBK geometric features (bishop color parity)

data/raw/{endgame}/full.csv        — gitignored, regenerate with build_dataset.py
data/processed/{endgame}/features.csv — gitignored, regenerate with build_features.py
logs/{endgame}_classifier/         — gitignored, contains model.joblib + tree.txt
storage/syzygy/3_4_5/              — gitignored, Syzygy .rtbw files needed locally

study/    — gitignored personal study notes with Q&A for chess/ML/professor audiences
```

## Run order (any endgame)

```bash
source venv/bin/activate
bash scripts/{endgame}/run_pipeline.sh
# or step by step:
python scripts/{endgame}/build_dataset.py   # ~75 min for 4-piece (parallelized)
python scripts/{endgame}/build_features.py  # ~5 min (parallelized, 2M sample)
python scripts/{endgame}/train_classifier.py
python scripts/{endgame}/validate.py
```

## Analysis scripts

```bash
python scripts/analysis/feature_importance.py   # top features per endgame
python scripts/kpk/compression_analysis.py      # compression ratio vs raw tablebase
```

## Key design decisions

- `min_samples_leaf=50` — forces rules that generalize, prevents fitting rare exceptions
- `max_depth=None` — depth is a result (finding), not a constraint
- Syzygy WDL only (no DTZ) — halves probe time, sufficient for WDL classification
- 2M position sample for 4-piece endgames — full 24.5M would OOM on 16GB
- Symmetric pieces enumerated with `p1 < p2` constraint to avoid duplicates
