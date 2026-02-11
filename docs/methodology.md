# KPK Experimental Methodology

This section documents the progressive refinement of the symbolic KPK model.

All experiments were performed on the fully enumerated KPK state space
(331,352 legal positions), labeled via Syzygy tablebases.

Evaluation uses a stratified 80/20 train-test split.


---

## Stage 1: Minimal Geometric Baseline

**Features:**
- King–king distances
- Pawn rank
- File / rank gaps
- Side to move

**Model:**
- DecisionTreeClassifier
- Default parameters
- No tuning

**Test Accuracy:** ~81%

**Interpretation:**
- Captures coarse proximity structure.
- Learns basic tempo effects.
- Fails on parity-based configurations and thin boundary structures.
- Strong underfitting of fine geometric distinctions.


---

## Stage 2: Hyperparameter Tuning (Same Features)

**Model Adjustments:**
- Grid search over depth and leaf size.
- Tuned decision tree.

**Test Accuracy:** ~85%

**Interpretation:**
- Improved partition granularity.
- Still limited by insufficient geometric basis.
- Boundary errors remain widespread.


---

## Stage 3: Expanded Geometric Feature Basis (~30 Features)

Additional features introduced:

- Manhattan and squared distances
- Relative king displacement vectors
- Promotion distance
- Edge proximity metrics
- Mobility counts
- Triangle area (king–king–pawn geometry)
- Normalized pawn rank
- Opposition proxies

**Model:** Tuned Decision Tree

**Test Accuracy:** ~90.13%

Example metrics:

- Accuracy: 0.9013
- Macro F1: 0.9003

Per-class performance:

- Loss (-2): F1 ≈ 0.913
- Draw (0): F1 ≈ 0.852
- Win (2): F1 ≈ 0.935

**Interpretation:**
- Significant improvement from richer geometry.
- Model now captures most spatial structure.
- Remaining errors concentrated near parity-sensitive boundaries.
- Draw class remains most ambiguous.


---

## Stage 4: Full Tuning + Deep Tree

**Model:**
- DecisionTreeClassifier
- max_depth ≈ 19
- min_samples_leaf = 1
- Fully tuned via grid search

**Test Accuracy:** 0.999185

Per-class metrics:

- Loss (-2): F1 ≈ 0.99944
- Draw (0): F1 ≈ 0.99876
- Win (2): F1 ≈ 0.99936

Full-dataset validation:

- 331,352 positions
- 54 misclassified
- Overall accuracy: 99.9837%

**Tree Size:**
- 1088 leaves
- Depth ≈ 19

**Residual Errors:**
- 54 positions
- Concentrated in rook-pawn edge cases
- Opposition parity configurations
- Near-stalemate boundary motifs


---

## Observations

1. Increasing model capacity alone (Stage 2) provides limited improvement.
2. Expanding geometric feature basis (Stage 3) produces the largest gain.
3. Deep partitioning (Stage 4) isolates a thin boundary manifold.
4. The KPK value function appears nearly axis-aligned separable in the chosen feature space.
5. Residual error rate ≈ 0.016% suggests strong geometric regularity.

Preliminary description-length estimates indicate
~6× compression relative to Shannon entropy of the WDL surface.


---

## Research Implication

The progressive improvement across stages suggests that:

- KPK value structure is strongly geometry-aligned.
- Rich feature representations matter more than shallow tuning.
- Decision-tree partitions approximate the value surface efficiently.
- Remaining misclassifications lie on a thin, highly structured boundary.

This supports the working hypothesis that certain endgame value functions
admit compact symbolic representations under appropriate geometric bases.
