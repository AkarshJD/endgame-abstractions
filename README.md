# Endgame Abstractions

A research framework for extracting human-interpretable geometric and
algorithmic\
principles of perfect chess endgame play using tablebases and machine
learning.

This project studies how optimal play emerges from spatial structure,\
with the goal of discovering minimal coordinate systems and symbolic
rules\
underlying solved endgames.

## Why This Project

Modern chess engines are superhuman but opaque.\
Endgame tablebases provide perfect play, but their internal structure is
largely uninterpretable.\
Human players, in contrast, reason using symbolic and geometric
concepts.

This project aims to bridge that gap by extracting interpretable
structure from solved endgames.

## Research Goal

To characterize perfect chess endgame play as a **low-dimensional,
interpretable dynamical system** over state space.

The aim is to identify:

-   Invariants\
-   Attractors\
-   Symbolic laws\
-   Minimal coordinate systems

governing optimal play.

## Overview

This project reverse-engineers that structure by:

-   Enumerating solved positions\
-   Extracting geometric features\
-   Learning symbolic rules\
-   Analyzing emergent concepts (opposition, zugzwang, triangulation,
    tempo)

Starting with simple endgames (KPK), the framework incrementally scales
to higher material.

## Repository Structure

``` bash
endgame_abstractions/
├── AUTHORS.md
├── LICENSE
├── README.md
├── checkpoints/
├── configs/
├── data/
├── docs/
├── experiments/
├── logs/
├── models/
├── notebooks/
├── outputs/
├── papers/
├── scripts/
├── src/
├── storage/
├── tests/
├── pyproject.toml
├── requirements.txt
└── venv/
```

Large binary data and tablebases are excluded from version control.

## Installation

Clone repository:

``` bash
git clone <repository-url>
cd endgame_abstractions
```

Create virtual environment:

``` bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
pip install -e .
```

## Syzygy Tablebase Setup (3--4--5 Pieces)

Create directory:

``` bash
mkdir -p storage/syzygy/3_4_5
cd storage/syzygy/3_4_5
```

Download from:

https://tablebase.lichess.ovh/tables/standard/

Required folders:

-   3-4-5-wdl\
-   3-4-5-dtz

Place all `.rtbw` and `.rtbz` files into:

``` bash
storage/syzygy/3_4_5/
```

These files are intentionally ignored by Git.

## KPK Research Pipeline (Baseline)

Run full pipeline:

``` bash
export PYTHONPATH=$(pwd)/src

python scripts/build_kpk_dataset.py
python scripts/build_kpk_features.py
python scripts/train_kpk_tree.py
python scripts/print_kpk_tree.py
```

### Pipeline stages:

1.  Enumerate legal KPK positions
2.  Query Syzygy tablebase
3.  Extract geometric features
4.  Train symbolic model
5.  Print learned rules

Output location:

``` bash
data/processed/kpk/
```

## KPK Baseline Results

The KPK endgame (King + Pawn vs King) has been fully enumerated and
labeled via Syzygy.

Current baseline symbolic model:

-   \~30 geometric features
-   Decision tree (max depth ≈ 19)
-   1088 leaves
-   331,352 legal positions
-   99.9837% full-dataset accuracy
-   54 boundary misclassifications

Error cases are concentrated in structurally thin boundary regions
(rook-pawn edge cases, opposition parity configurations, near-stalemate
motifs).

Preliminary compression analysis suggests approximately **6×
reduction**
relative to the Shannon entropy of the WDL surface.

This supports the working hypothesis that certain endgame value
functions
exhibit strong geometric regularity.

## Position Analysis

Analyze individual positions:

``` bash
python src/endgame/kpk_analyzer.py
```

Edit the FEN string inside the script.

## Research Methodology

The project follows a bottom-up discovery approach:

1.  Exhaustive state enumeration
2.  Perfect-information labeling
3.  Feature construction
4.  Symbolic learning
5.  Rule compression
6.  Theoretical interpretation

Chess knowledge is not manually encoded.
All abstractions are learned from solved data.

## Machine Learning Approach

For each endgame:

-   State space is fully enumerated
-   Each position is labeled (WDL, DTZ)
-   Geometric features are computed
-   Decision trees and rule learners are trained
-   Learned rules are simplified and interpreted

This yields symbolic approximations of perfect play.

## Current Status

### Implemented:

-   Syzygy integration
-   KPK dataset generation
-   Geometric feature engineering
-   Decision tree learning
-   Full-dataset validation
-   Misclassification boundary analysis
-   Compression estimation

### In Progress:

-   Feature ablation studies
-   Boundary manifold characterization
-   Visualization layer
-   Generalization testing (KRK, KPPK)

### Planned:

-   KRKP, KBNK, KQK
-   Multi-piece abstraction
-   Graph-theoretic modeling
-   Neuro-symbolic systems
-   Generative state models

## License

See LICENSE file.

## Author

Akarsh J D

Status: Active Research

## Citation and Status

This project is an active research program and is currently in
pre-publication stage.

A formal citation will be provided after peer-reviewed or preprint
publication.

If you use this code in academic work prior to publication,
please reference the repository URL and contact the author.

## Acknowledgments

This project makes use of Syzygy endgame tablebases
developed by Ronald de Man and distributed by Lichess.

Tablebases are obtained from:

https://tablebase.lichess.ovh/

We thank the Lichess community for providing open access
to high-quality endgame data for research purposes.
