# Feature Registry

## Purpose

The Feature Registry defines the **coordinate system** used to represent chess endgame
positions in this project.

Rather than encoding named chess concepts (e.g., opposition, zugzwang, key squares),
the framework represents positions using **low-level geometric and temporal coordinates**.
Higher-level structure is expected to **emerge from combinations of these coordinates**
during learning and symbolic analysis.

The registry provides a single source of truth for how game states are represented and
ensures that all features are:

- Explicitly defined
- Reproducible
- Interpretable at the geometric level
- Comparable across experiments
- Traceable through version control


## Design Principles

### 1. Identifier Stability

Each feature is assigned an **opaque, stable identifier** of the form `f0001`, `f0002`, etc.

- Identifiers must never be reused
- Identifiers carry no semantic meaning
- All experimental results refer to features by ID, not by name

This guarantees comparability across experiments, models, and time, even as the registry evolves.


### 2. Geometry-First Representation

All features describe **geometric or temporal properties** of the board state.

Feature geometries include:

- **Chess-metric geometry** (king-move distance)
- **Projected geometry** (rank-only or file-only projections)
- **Topological relations** (alignment predicates)
- **Directed geometry** (pawn advancement toward promotion)
- **Temporal dimensions** (side to move)

No strategic concepts, evaluation labels, or tablebase knowledge are explicitly encoded.


### 3. Separation of Representation and Theory

The registry defines **coordinates**, not **concepts**.

Classical chess ideas (e.g., opposition or triangulation) are understood as
*human interpretations* of patterns that may arise along these coordinates.
They are not encoded directly.

This separation allows learned models to rediscover, recombine, or reject
human theory without constraint.


## Feature Classification

### Primitive vs. Derived

- **Primitive features** are computed directly from the board state
- **Derived features** are computed from other features

This distinction supports ablation studies, minimal basis analysis,
and feature compression.


### Interpretability Level

Each feature includes an `interpretability_level` annotation:

- **0 — Opaque**: Raw or low-level coordinate with no common human naming
- **1 — Derived geometric**: Interpretable geometric quantity without a standard name
- **2 — Classically named**: Commonly associated with named concepts in chess literature

This annotation is **descriptive only**.  
It does not imply the injection of strategy or prior knowledge.


## Why This Matters

A formal feature registry enables:

- Analysis of which geometric dimensions are sufficient for perfect play
- Comparison of learned abstractions across different endgames
- Identification of redundant or compressible coordinates
- Extraction of symbolic rules from learned decision structures
- Long-term study of minimal coordinate systems governing optimal play

By fixing the representation layer, the project can study **emergence, compression,
and explainability** without confounding changes in state encoding.


## Usage and Governance

- All feature extraction code must correspond to registry entries
- Experimental logs must record feature IDs used
- New features must be added to the registry before use
- Registry changes should be committed independently of experimental results

This document is a **living specification** of the state-space representation and
forms the foundation for all downstream learning and analysis.
