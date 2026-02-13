# Architecture: Endgame Abstractions

This document describes the structural organization and design principles
of the Endgame Abstractions project.

The system is designed to:

- Support multiple solved endgames
- Maintain a unified geometric feature registry
- Enable symbolic model training (classification and regression)
- Support compression and abstraction experiments
- Preserve reproducibility and experiment traceability


## High-Level System Overview

Each endgame follows a consistent experimental pipeline:

1. Enumerate legal positions
2. Query Syzygy tablebase (WDL / DTZ)
3. Extract geometric features
4. Train symbolic models
5. Evaluate and log results
6. Analyze compression and structural patterns


## Repository Structure

```bash
endgame_abstractions/
├── AUTHORS.md
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── raw/                # Raw enumerated tablebase positions
│   ├── processed/          # Feature-extracted datasets
│   └── samples/
│
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── roadmap.md
│
├── logs/
│   ├── kpk/
│   └── krk/
│
├── scripts/
│   ├── kpk/
│   │   ├── build_dataset.py
│   │   ├── build_features.py
│   │   ├── train_tree.py
│   │   ├── tune_tree.py
│   │   ├── train_regression.py
│   │   ├── validate.py
│   │   └── analyze.py
│   │
│   └── krk/
│       ├── build_dataset.py
│       ├── build_features.py
│       └── train_tree.py
│
├── src/
│   └── endgame/
│       ├── features/
│       │   ├── feature_registry.yaml
│       │   ├── kpk_geom.py
│       │   └── krk_geom.py
│       │
│       ├── tb/
│       ├── learning/
│       ├── evaluation/
│       └── kpk_analyzer.py
│
├── storage/
│   └── syzygy/
│       └── 3_4_5/
│
└── tests/
```

Large binary data (Syzygy tablebases) and experiment logs are excluded from version control.


## Design Principles

### Per-Endgame Modularity

Each endgame (e.g., KPK, KRK) has its own pipeline directory under:

    scripts/<endgame>/

This isolates:

- Enumeration logic
- Feature extraction
- Model training
- Evaluation
- Regression experiments


### Unified Feature Registry

All geometric features are defined centrally in:

    src/endgame/features/feature_registry.yaml

The registry:

- Defines primitive geometric coordinates
- Defines derived features
- Maintains stable, opaque feature IDs
- Supports extension without breaking prior experiments


### Separation of Concerns

- `scripts/` → experiment orchestration
- `src/endgame/features/` → geometric feature definitions
- `src/endgame/tb/` → tablebase integration
- `logs/` → experiment outputs
- `data/` → datasets


### Experiment Logging

Each training run creates a timestamped directory under:

    logs/<endgame>/

Containing:

- Model parameters
- Metrics
- Tree export
- Metadata (git hash, dataset size)

This ensures reproducibility and auditability.


### Extensibility

To add a new endgame:

1. Implement geometry extractor in `src/endgame/features/`
2. Create pipeline under `scripts/<endgame>/`
3. Enumerate dataset
4. Train symbolic model
5. Log results

No existing pipeline needs modification.
