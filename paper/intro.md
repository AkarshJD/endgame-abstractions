# Introduction

## The Fidelity Problem in Explainable AI

Explainability methods for machine learning are evaluated on their *fidelity* to
the underlying model: does the explanation faithfully represent why the model
makes its predictions? A deeper requirement, rarely satisfiable in practice, is
fidelity to the *ground truth*: when the model is wrong, can we explain *why* it
is wrong?

This distinction matters because fidelity failures have two causally distinct
sources that the XAI literature routinely conflates:

1. **Excluded-information failure.** The model cannot access information the
   correct label depends on. Richer features or additional observables would
   resolve this; the model class is not the bottleneck.

2. **Intrinsic boundary sensitivity.** The true decision boundary flips under
   minimal perturbation even when the model has access to all relevant
   information. Look-ahead search or a richer model class is required; the
   features themselves are insufficient.

Separating these two mechanisms in real domains is impossible: there is no oracle
to compare against. A chess endgame tablebase is a perfect oracle. It provides
ground-truth outcomes for every legal position, making causal attribution of
fidelity failure tractable.

## The Oracle Setting

We use Syzygy endgame tablebases [CITE: Ronald de Man] as a perfect oracle over
the space of low-piece chess endgames. For every legal position, the tablebase
specifies the exact outcome under optimal play: Win, Draw, or Loss for the side
to move.

We train decision tree classifiers on hand-engineered geometric features — piece
distances, corner proximity, spatial parity — to predict WDL (Win/Draw/Loss)
without look-ahead search. Prior work established that static geometric features
achieve >99.5% per-class recall (fidelity) on WIN and LOSS, while DRAW fidelity
falls to approximately 97.4% [CITE: self]. All misclassifications occur at the
Draw boundary — zero Win↔Loss errors across every endgame tested. This is a
striking empirical pattern. The present paper asks: what causes the draw-boundary
gap?

## Why Testing the Hypothesis Carefully Is Non-Trivial

The natural candidate is the **50-move rule**: if 50 moves pass without a pawn
move or capture, the game is drawn by claim. Positions that are theoretically
winning but require more than 50 moves to convert are called *cursed wins*
(raw Syzygy WDL code = +1); their correct outcome depends on the halfmove clock,
which is absent from the geometric feature set. If draw-boundary errors
concentrate on cursed positions, the cause is excluded information (the clock),
not the geometric complexity of the boundary.

Testing this hypothesis naively is dangerous. One approach — adding
Distance-to-Zeroing (DTZ), a Syzygy-derived feature encoding the clock's
proximity, to the model — is **circular**: WDL and DTZ are both outputs of the
same oracle. Predicting one from a function of the other trivially produces high
attributed shares without any causal content. We demonstrate this leakage
explicitly (Section 3.2) and propose the correct test.

## Contributions

1. **Per-class fidelity audit.** We establish baseline per-class recall (fidelity)
   for three endgames (KBNK, KRK, KPK) under matched-pair cross-validation,
   confirming draw-boundary concentration and ruling out Win↔Loss errors.

2. **Leakage exhibit.** We show that adding DTZ as a model feature is a
   methodological circular reference — both WDL and DTZ are Syzygy-derived —
   and demonstrate the telltale signature: a flat 100% excluded-info share across
   structurally different endgames.

3. **Cursed-position enumeration.** Using raw WDL codes (±1 = clock-dependent,
   ±2 = clean) as a *label property* — never as a model feature — we enumerate
   cursed and blessed positions across all three endgames. This is the correct,
   non-circular test of the 50-move-rule hypothesis.

4. **Falsification.** All three endgames contain zero cursed or blessed positions
   (KBNK: 24.5M probed; KRK: 399K probed; KPK: DTZ range [−20, 19]). The
   50-move-rule mechanism is structurally absent from these datasets. All
   draw-boundary errors are geometric.

5. **Learnable-vs-intrinsic test.** A matched-pair Random Forest comparison
   (200 trees, identical CV folds) shows DRAW recall improves by +0.0064 while
   LOSS recall degrades by −0.0051. The coupled trade-off indicates a calibration
   shift, not boundary learning. Approximately 75% of draw-boundary errors are
   common to both classifiers and intrinsic to the geometric feature set.

The draw-boundary fidelity gap is not caused by the 50-move rule and is not
resolved by a stronger model class. It reflects structural limits of the geometric
feature representation at the draw boundary.

The method generalises: any domain with a perfect oracle (SAT solvers, planning
domains with verified solutions) can apply this decomposition to audit fidelity
failure and detect circular leakage in feature selection.

## Paper Organisation

Section 2 describes the experimental setup. Section 3 presents the
decomposition: baseline fidelity, the leakage exhibit, the cursed enumeration,
and the learnable test. Section 4 discusses generalisability and open questions.
