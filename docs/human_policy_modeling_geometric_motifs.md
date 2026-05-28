# Human Policy Modeling via Geometric Motif Abstractions

## Overview

This document outlines a research direction extending symbolic
compression of deterministic chess endgames toward modeling human move
selection using interpretable geometric features.

The central idea is:

Instead of modeling perfect-play value functions alone, we model human
policy as a function of structured geometric motifs.

The goal is not to replicate engine strength, but to understand whether
low-complexity motif representations can approximate human
decision-making patterns.

## Background

Current engine paradigms rely on:

-   Deep tree search
-   Minimax or alpha-beta pruning
-   Neural evaluation networks
-   Large-scale self-play training

Humans, however, appear to rely more heavily on:

-   Pattern recognition
-   Structural motifs
-   Distance relationships
-   Parity and opposition awareness
-   Candidate move narrowing

This project investigates whether symbolic geometric feature bases can
approximate human move selection without brute-force search.

## Formal Problem Statement

Let:

-   x = position
-   φ(x) = geometric feature vector
-   m = move played by a human

We aim to learn:

P(m \| φ(x))

rather than:

BestMove(x) = argmax over search tree

The objective is to determine whether a low-complexity decision tree
over φ(x) can approximate human move distributions.

## Feature Representation

The feature basis is inherited from the deterministic KPK compression
framework.

Examples include:

-   King--king Chebyshev distance
-   King--pawn distance
-   Pawn rank
-   File proximity to edge
-   Opposition parity
-   Promotion square control
-   Tempo-sensitive parity indicators

These features form a motif-based structural encoding of the position.

## Methodology

### Dataset Construction

1.  Collect human endgame games from Lichess or similar database.
2.  Filter for positions matching target material class (e.g., KPK).
3.  Extract move played by human.
4.  Compute geometric feature vector φ(x).

### Model

Train a decision tree classifier:

Input: φ(x)\
Output: move class or move cluster

Evaluate using:

-   Train/test split
-   Cross-validation
-   Stratified sampling by player rating

## Experimental Axes

### Rating Stratification

Partition dataset into rating bins:

-   1200--1500
-   1500--2000
-   2000--2400
-   2400+

Hypothesis:

Stronger players may exhibit:

-   Lower entropy in move selection
-   More structured motif usage
-   Shallower but more consistent tree structure

### Baselines

Compare against:

-   Random move baseline
-   Frequency-based baseline
-   Shallow search baseline
-   Neural policy baseline (optional)

## Evaluation Metrics

-   Accuracy
-   Top-k accuracy
-   Entropy of predicted move distribution
-   Tree depth
-   Number of leaves
-   Feature importance stability

Interpretability metrics:

-   Average rule length
-   Repetition of motif predicates
-   Boolean factorization potential

## Research Questions

1.  Can human move selection in deterministic endgames be approximated
    using a low-complexity geometric feature basis?
2.  Does policy complexity vary with player rating?
3.  Are specific motif clusters predictive of player strength?
4.  Does structural compression correlate with expertise?

## Interpretability Angle

Decision trees provide explicit Boolean rule paths.

Each leaf corresponds to:

If motif A AND motif B AND NOT motif C → Move X

This allows:

-   Direct inspection of motif usage
-   Extraction of human-style rules
-   Comparison across rating groups

This aligns with cognitive modeling rather than engine performance
modeling.

## Expected Outcomes

Possible outcomes include:

1.  Human policy compresses well under geometric basis.
2.  Human policy remains noisy and requires deeper trees.
3.  Strong players show higher structural consistency.
4.  Weak players exhibit fragmented motif structure.

All outcomes are informative.

## Scope Control

This project is separate from:

-   Perfect-play compression experiments
-   n×n scaling experiments
-   LLM explanation pipelines
-   General cognitive theory claims

This is a contained empirical study:

Geometric feature basis → human move modeling → interpretable policy
structure.

## Long-Term Extension (Optional)

If initial experiments are successful:

-   Expand to KRK or small multi-piece endgames.
-   Compare motif-based trees against shallow search heuristics.
-   Explore clustering of players by learned policy trees.
-   Integrate LLM to verbalize extracted motif rules.

These extensions are optional and contingent on initial results.

## Research Positioning

This project sits at the intersection of:

-   Symbolic machine learning
-   Interpretability
-   Cognitive modeling
-   Deterministic game analysis
-   Policy compression

It does not attempt to outperform engines.

It attempts to measure structural compressibility of human
decision-making under motif-based representation.
