# KPK Experimental Methodology

This section documents the progressive refinement of the symbolic KPK model.

All experiments were performed on the fully enumerated KPK state space
(331,352 legal positions), labeled via Syzygy tablebases.

Evaluation uses a stratified 80/20 train-test split.


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


## Stage 2: Hyperparameter Tuning (Same Features)

**Model Adjustments:**
- Grid search over depth and leaf size.
- Tuned decision tree.

**Test Accuracy:** ~85%

**Interpretation:**
- Improved partition granularity.
- Still limited by insufficient geometric basis.
- Boundary errors remain widespread.


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


## Observations

1. Increasing model capacity alone (Stage 2) provides limited improvement.
2. Expanding geometric feature basis (Stage 3) produces the largest gain.
3. Deep partitioning (Stage 4) isolates a thin boundary manifold.
4. The KPK value function appears nearly axis-aligned separable in the chosen feature space.
5. Residual error rate ≈ 0.016% suggests strong geometric regularity.

Preliminary description-length estimates indicate
~6× compression relative to Shannon entropy of the WDL surface.


## Research Implication

The progressive improvement across stages suggests that:

- KPK value structure is strongly geometry-aligned.
- Rich feature representations matter more than shallow tuning.
- Decision-tree partitions approximate the value surface efficiently.
- Remaining misclassifications lie on a thin, highly structured boundary.

This supports the working hypothesis that certain endgame value functions
admit compact symbolic representations under appropriate geometric bases.


## KPK Regression Results (DTZ)

In addition to WDL classification, regression models were trained to
predict DTZ (Distance to Zeroing move).

Experiments were performed separately by structural regime:

### Winning Positions (WDL = 2)

- Samples: ~125k
- MAE ≈ 0.22
- R² ≈ 0.83
- Tree depth ≈ 10
- Leaves ≈ 200

Interpretation:
- Winning race geometry exhibits moderate compressibility.
- DTZ surface is not perfectly axis-aligned.
- Structural complexity remains within win manifold.


### Losing Positions (WDL = -2)

- MAE ≈ 0.25
- R² ≈ 0.83
- Tree depth ≈ 10
- Leaves ≈ 200

Interpretation:
- Symmetric structural behavior to winning regime.
- Similar geometric complexity profile.


### Draw Positions (WDL = 0)

- MAE = 0.0
- R² = 1.0
- Tree depth = 0
- Leaves = 1

Interpretation:
- DTZ constant in draw regime.
- Structurally trivial value surface.
- No geometric variation to model.


### Regime Observation

Different WDL regimes exhibit different structural compressibility:

- Draw manifold: trivial
- Win/Loss manifolds: moderately compressible
- Classification surface: highly compressible

This suggests that value surfaces decompose into structurally distinct
submanifolds.


## KRK Baseline Regression Results

KRK experiments model DTZ directly (regression only).

Dataset:
- ~399k legal KRK positions
- Fully labeled via Syzygy

### Baseline Tree

- MAE ≈ 2.09
- R² ≈ 0.981
- Depth ≈ 10
- Leaves ≈ 658

### Tuned Tree

- MAE ≈ 1.68
- R² ≈ 0.986
- Depth ≈ 26
- Leaves ≈ 17,959

After feature refinement:

- MAE ≈ 1.51
- R² ≈ 0.9898
- Depth ≈ 10
- Leaves ≈ 706

### Interpretation

- KRK DTZ surface is significantly smoother than KPK race surface.
- Confinement geometry dominates structural variation.
- Moderate-depth trees achieve high explanatory power.
- Regression is more stable than KPK race regime.

---

## Comparative Structural Insight

| Endgame | Task | Structural Complexity | Compressibility |
|----------|------|----------------------|----------------|
| KPK | WDL Classification | Thin boundary manifold | Very High |
| KPK | DTZ Regression (Win/Loss) | Race geometry | Moderate |
| KRK | DTZ Regression | Confinement geometry | High |

These observations reinforce the hypothesis that:

- Different endgames exhibit distinct geometric regimes.
- Value-function complexity varies by structural mechanism.
- Symbolic compression depends on underlying geometry.
