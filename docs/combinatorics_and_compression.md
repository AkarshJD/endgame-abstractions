# Combinatorics and Symbolic Compression in KPK

Status: Working Theoretical Framework

This document captures hypotheses and structural interpretations
emerging from empirical KPK results.
General claims are conjectural pending validation
on additional endgames.

The central thesis is:

> The KPK value function exhibits substantial symbolic compression 
under a geometric feature basis, suggesting structured low effective 
complexity relative to the raw state space because its win/draw/loss 
surface is geometrically structured
> and low-dimensional relative to the raw state space.


# 1. Raw State Space Combinatorics

## 1.1 Naïve Enumeration

Ignoring legality:

- White King: 64 squares
- Black King: 64 squares
- Pawn: 64 squares

Naïve upper bound:

64^3 = 262,144 positions

However, constraints apply:

- Kings cannot overlap
- Kings cannot be adjacent
- Pawn cannot occupy rank 1 or 8
- Side to move doubles state space
- Illegal check positions removed

After legality filtering and normalization,
the exhaustive KPK dataset contains:

331,352 legal positions

This is the full supervised ground-truth
value surface derived from Syzygy.


## 1.2 Symmetry Reduction

The chessboard admits automorphisms:

- Horizontal reflection
- Vertical reflection
- 180° rotation

Under canonicalization:

- Pawn file symmetry reduces equivalent states
- King configurations map into equivalence classes

This implies the effective geometric degrees of freedom
are strictly lower than raw combinatorics suggests.

Key Insight:

The value function is invariant under board symmetries,
implying compressibility through canonical coordinate choice.


# 2. Degrees of Freedom Analysis

A KPK position is not truly 3 × 2 spatial degrees of freedom.

Constraints induce coupling:

- King adjacency constraints
- Pawn directionality (asymmetric forward motion)
- Promotion boundary conditions
- Opposition parity

Empirically, value transitions are governed by:

- Relative king distances
- Pawn advancement depth
- Opposition structure
- Edge constraints (rook pawn effect)

This suggests the state space lies on a structured manifold,
not in a uniform 3-piece Cartesian cube.


# 3. Geometric Feature Construction

## 3.1 Feature Philosophy

Features are constructed as geometric invariants:

- King–king distance
- King–pawn distance
- Pawn rank
- File proximity to edge
- Opposition parity indicators
- Chebyshev and Manhattan metrics

These are coordinate projections of spatial relationships.


## 3.2 Axis-Aligned Separability Hypothesis

Empirical observation:

The KPK value surface is nearly separable
along a small number of geometric axes.

Evidence:

- Decision tree achieves 99.9837% accuracy
- Only 54 boundary misclassifications
- Shallow depth relative to state space size

Interpretation:

The WDL function is almost axis-aligned
in the constructed geometric coordinate system.

This is a strong compression signal.


# 3.3 Logical Feature Algebra

Geometric primitives alone do not fully explain compression.

A second structural layer exists:

Boolean composition of geometric predicates.

Let primitive predicates be defined as:

P1 := (king_distance <= 1)
P2 := (pawn_rank >= 6)
P3 := (opposition == true)
P4 := (pawn_file_is_edge)
P5 := (promotion_square_controlled)

These are binary-valued geometric conditions.

The decision tree does not merely split on raw numeric features.
It implicitly constructs Boolean combinations:

- P1 AND P3
- P2 AND NOT P5
- P3 XOR tempo_parity
- P4 AND (P2 OR P3)

Thus the tree is implementing a Boolean circuit
over geometric primitives.


# 3.4 Boolean Basis and Algebraic Compression

Consider the win condition as a Boolean function:

W = f(P1, P2, ..., Pk)

The learned tree approximates f via recursive partitioning.

However, many branches represent repeated logical motifs:

Example:

(P2 AND P3) OR (P2 AND P4)

This can be factorized:

P2 AND (P3 OR P4)

Tree representation:
    two separate branches

Algebraic representation:
    one factored condition

This suggests additional compression is possible
through Boolean algebra simplification.


# 3.5 XOR and Parity Structures

Opposition and tempo introduce parity-sensitive structure.

These often behave like XOR relations:

win_condition ≈ (opposition XOR tempo_parity)

or

draw_condition ≈ NOT (king_in_front XNOR pawn_rank_parity)

Such relationships are not axis-aligned thresholds.
They are relational constraints.

Standard decision trees approximate XOR via multiple splits,
which inflates depth.

Recognizing XOR structure explicitly
can reduce symbolic length.


# 3.6 Feature Coupling and Clustering

Primitive features can be grouped by geometric theme:

Distance Cluster:
- king_distance
- king_pawn_distance
- Chebyshev metrics

Edge Cluster:
- pawn_file_edge
- promotion_square_file
- rook_pawn_indicator

Parity Cluster:
- opposition
- side_to_move
- pawn_rank_parity

Control Cluster:
- promotion_square_control
- king_in_front
- blockade_status

Instead of treating features independently,
we can define meta-predicates:

DistanceDominance := (king_distance <= king_pawn_distance)
EdgeConstraint := pawn_file_edge AND promotion_square_edge
ParityControl := opposition XOR tempo_parity

This converts flat features into structured logical units.


# 3.7 Toward Shorter Trees via Logical Factorization

Tree length inflation occurs when:

- Same predicate is repeated in multiple branches
- XOR-like conditions are decomposed into AND/OR splits
- Symmetric cases are represented redundantly

Potential compression strategy:

1. Extract all leaf paths as Boolean expressions.
2. Convert to Disjunctive Normal Form (DNF).
3. Apply Boolean minimization (e.g., Quine–McCluskey style).
4. Refactor common subexpressions.
5. Reconstruct minimal symbolic rule set.

Goal:

Minimize:

L(Boolean rule set)

rather than:

L(tree splits)


# 3.8 Interpretability Implication

Geometric features explain *what varies spatially*.

Boolean algebra explains *how those variations combine logically*.

Compression therefore occurs on two levels:

Level 1: Geometric coordinate reduction  
Level 2: Logical expression minimization  

The tree captures both implicitly.

Explicit extraction of logical structure
may reduce description length further
and produce more human-readable laws.


# 3.9 Research Direction

Open Questions:

- Does KPK admit a minimal Boolean basis of small size?
- Are parity relations inherently XOR-like?
- Can boundary manifold structure be described
  as low-degree Boolean polynomial?
- Does logical factorization reduce tree depth consistently?

Hypothesis:

The symbolic complexity of KPK is bounded not only
by geometric dimensionality,
but by low Boolean circuit complexity
over geometric predicates.


# 4. Description Length Perspective

## 4.1 Shannon Entropy Baseline

The WDL distribution defines a discrete random variable.

Let:

H(WDL) = Shannon entropy of the value surface

Raw encoding of 331,352 positions requires O(N) bits.


## 4.2 Symbolic Model Encoding

A decision tree partitions space into
rectangular regions.

Encoding requires:

- Feature index
- Threshold
- Leaf label

Empirically, tree representation length
is ~6× smaller than naïve encoding.


## 4.3 Minimum Description Length (MDL) Framing

Interpret the tree as hypothesis H
that compresses dataset D.

Total description length:

L(H) + L(D | H)

Since training error is near-zero:

L(D | H) ≈ boundary correction cost

The small residual error count (54)
implies boundary complexity is low.

This supports the hypothesis:

KPK has low algorithmic information
relative to its raw combinatorial size.


# 5. Boundary Manifold Geometry

Misclassifications cluster around:

- Rook pawn edge cases
- Parity-sensitive opposition
- Promotion timing races

This suggests:

The decision boundary is a thin,
structured manifold embedded in feature space.

Hypothesis:

Boundary dimension << ambient feature dimension.


# 6. Structural Compression Hypothesis

Define structural compressibility as:

Existence of a low-complexity symbolic model
that approximates the perfect-play value function
with negligible residual error.

KPK satisfies:

- Small tree depth
- Low misclassification count
- Stable feature importance
- High symmetry alignment

Open Question:

Is KPK uniquely compressible,
or representative of small-material endgames?


# 7. Scaling Considerations

As material increases:

- State space grows combinatorially
- Interaction terms increase
- Geometric invariants multiply
- Boundary curvature may increase

Key Research Question:

Does description complexity grow sublinearly
relative to state space size?


# 8. Toward a General Theory

Conjecture:

Perfect-play value functions in finite deterministic games
often lie on low-dimensional structured manifolds
when expressed in appropriate invariant coordinates.

Future directions:

- Formal boundary dimension estimation
- Algebraic characterization of invariants
- Topological analysis of win/draw partitions
- Algorithmic information bounds


# 9. Summary

KPK demonstrates:

- Large raw combinatorial state space
- Strong geometric structure
- Near-axis-aligned decision boundaries
- Significant symbolic compressibility

The compression is geometric, not statistical.

This document formalizes the theoretical foundation
for extending symbolic compression analysis
to richer endgames.
