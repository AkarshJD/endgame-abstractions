# Endgame Abstractions

A research framework for extracting human-interpretable geometric and
algorithmic principles of perfect chess endgame play using tablebases
and machine learning.

This project studies how optimal play emerges from spatial structure,
with the goal of discovering minimal coordinate systems and symbolic
rules underlying solved endgames.


## Why This Project

Modern chess engines are superhuman but opaque. Endgame tablebases
provide perfect play, but their internal structure is largely
uninterpretable. Human players, in contrast, reason using symbolic and
geometric concepts.

This project aims to bridge that gap by extracting interpretable
structure from solved endgames.


## Research Goal

To characterize perfect chess endgame value functions as
low-dimensional, interpretable dynamical systems over discrete state space.

The aim is to identify:

- Invariants
- Attractors
- Symbolic laws
- Minimal coordinate systems

governing optimal play.


## Overview

This project reverse-engineers that structure by:

- Enumerating solved positions
- Extracting geometric features
- Learning symbolic rules
- Analyzing emergent concepts (opposition, zugzwang, triangulation, tempo)

Starting with simple endgames (KPK), the framework incrementally scales
to higher material.


## Repository Structure

For detailed architectural layout and module organization, see:

docs/architecture.md


## Installation

Clone repository:

```bash
git clone <repository-url>
cd endgame_abstractions
```

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```


## Syzygy Tablebase Setup (3-4-5 Pieces)

Create directory:

```bash
mkdir -p storage/syzygy/3_4_5
cd storage/syzygy/3_4_5
```

Download from:

https://tablebase.lichess.ovh/tables/standard/

Required folders:

- 3-4-5-wdl
- 3-4-5-dtz

Place all `.rtbw` and `.rtbz` files into:

```bash
storage/syzygy/3_4_5/
```

These files are intentionally ignored by Git.


## KPK Research Pipeline

Run full pipeline:

```bash
export PYTHONPATH=$(pwd)/src

python scripts/kpk/build_dataset.py
python scripts/kpk/build_features.py
python scripts/kpk/train_tree.py
python scripts/kpk/print_kpk_tree.py
```

Validate trained model:

```bash
python scripts/kpk/validate.py
```

Output location:

```bash
data/processed/kpk/
```


## KPK Baseline Results

The KPK endgame (King + Pawn vs King) has been fully enumerated and
labeled via Syzygy.

Current symbolic model:

- ~30 geometric features
- Decision tree (max depth ≈ 19)
- 1088 leaves
- 331,352 legal positions
- 99.9837% full-dataset accuracy
- 54 boundary misclassifications

Error cases are concentrated in structurally thin boundary regions
(rook-pawn edge cases, opposition parity configurations, near-stalemate
motifs).

Preliminary analysis suggests significant geometric compression
relative to naive entropy estimates of the WDL surface.

This supports the working hypothesis that certain endgame value
functions exhibit strong geometric regularity.


## KPK Regression Experiments (DTZ)

In addition to WDL classification, regression models are trained to
predict DTZ (Distance to Zeroing move) for winning positions.

Example:

```bash
python scripts/kpk/train_regression.py
```

Results indicate:

- Strong compression within winning regimes
- Different structural compressibility across win / loss / draw regions
- Regime-dependent geometric structure


## Position Analysis

Analyze individual positions:

```bash
python src/endgame/kpk_analyzer.py
```

Edit the FEN string inside the script.


## Research Methodology

The project follows a bottom-up discovery approach:

1. Exhaustive state enumeration
2. Perfect-information labeling
3. Feature construction
4. Symbolic learning
5. Rule compression
6. Theoretical interpretation

Chess knowledge is not manually encoded. All abstractions are learned
from solved data.


## Machine Learning Approach

For each endgame:

- State space is fully enumerated
- Each position is labeled (WDL, DTZ)
- Geometric features are computed
- Decision trees and rule learners are trained
- Learned rules are simplified and interpreted

This yields compact symbolic approximations of perfect-play value functions.


## Current Status

### Implemented:

- Syzygy integration
- Modular per-endgame pipeline structure
- KPK dataset generation
- KPK WDL classification (99.98% full-dataset accuracy)
- KPK DTZ regression experiments
- KRK baseline modeling
- Experiment logging and reproducibility framework
- Misclassification boundary analysis
- Compression estimation

### In Progress:

- KRK symbolic modeling
- Feature ablation studies
- Boundary manifold characterization
- Generalization testing (KPPK, KRKP)

### Planned:

- KBNK, KQK
- Multi-piece abstraction
- Graph-theoretic modeling
- Neuro-symbolic systems
- Generative state models


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

If you use this code in academic work prior to publication, please
reference the repository URL and contact the author.


## Acknowledgments

This project makes use of Syzygy endgame tablebases developed by Ronald
de Man and distributed by Lichess.

Tablebases are obtained from:

https://tablebase.lichess.ovh/

We thank the Lichess community for providing open access to high-quality
endgame data for research purposes.
