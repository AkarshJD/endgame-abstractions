# Faithfulness Decomposition — Paper Workspace

**Branch:** `faithfulness-decomposition`
**Status:** Experiment v2 complete. RF learnable test running. Paper skeleton in progress.

---

## What This Paper Is

Explainability research evaluates whether a model's explanation is *faithful* to
its predictions. A harder question is whether the model is faithful to the
*ground truth*, and when it fails, *why*.

A chess endgame tablebase is a perfect oracle — one of the only domains where
this attribution can be made causally exact. We train decision tree classifiers
on geometric features to predict Win/Draw/Loss (WDL) without look-ahead search,
and ask: where does the model fail, and what causes those failures?

**Prior finding:** >99.5% per-class recall on WIN and LOSS; ~97.4% on DRAW.
Every misclassification involves the Draw boundary — zero Win↔Loss errors.

**This paper asks:** What causes the draw-boundary gap?

---

## The Experiment

### What We Actually Did (v2 — correct)

1. **Baseline fidelity** — 5-fold matched-pair CV on geometric model (no DTZ).
   Per-class recall (fidelity) for WIN / DRAW / LOSS separately.

2. **Cursed-position enumeration** — probe raw WDL codes (±1 = clock-dependent,
   ±2 = clean) for all positions in each endgame dataset. This is a *label
   property check*, not a feature. DTZ never enters the model.

3. **Draw-boundary error classification** — of the errors the model makes, how
   many land on cursed positions (raw WDL = ±1)? If many: the 50-move clock
   caused them. If zero: all errors are geometric.

4. **RF learnable test** (in progress) — train RandomForestClassifier on the
   same geometric features under matched-pair CV. If DRAW recall improves
   substantially: errors were model-class artifacts (decision tree underfitting).
   If recall stays flat: the boundary is intrinsically hard.

### What v1 Did Wrong (the leakage exhibit)

v1 added DTZ as an explicit feature and retrained. This is circular: WDL and
DTZ are both Syzygy outputs — predicting one from the other is like predicting
a number using a function of that number. v1 produced 100% "excluded-info share"
across all three endgames, a flat ceiling that signals leakage, not a finding.

---

## Confirmed Results

| Endgame | Total positions | Cursed/blessed |
|---|---|---|
| KBNK | 24,536,088 | **0** |
| KRK  | 399,112    | **0** |
| KPK  | 331,352    | **0** (DTZ range: [−20, 19]) |

**The 50-move-rule mechanism does not exist in any of these three endgames.**
All draw-boundary errors are geometric. The clock hypothesis is falsified.

---

## Outcomes (pending RF result)

**If RF DRAW recall >> DT DRAW recall (delta > ~0.01):**
> Boundary errors are model-class artifacts. The decision tree with
> `min_samples_leaf=50` underfits the draw-boundary geometry. The real
> contribution is the method for ruling out excluded-information failure
> and diagnosing representational underfitting using a ground-truth oracle.

**If RF DRAW recall ≈ DT DRAW recall (delta < ~0.003):**
> Boundary errors are intrinsically geometric. Even a much stronger model
> class cannot learn them from the same features. This is the oracle-grounded
> measurement of where static pattern recognition ends. The natural next step
> is feature expansion or neural representation — not model class.

---

## Endgames

| Endgame | Role | Draw mechanism |
|---|---|---|
| KBNK | Primary | Bishop-knight mating geometry; 50-move rule ruled out |
| KRK  | Replication | Stalemate only; no 50-move-rule involvement |
| KPK  | Replication | Pawn blockade / king opposition; DTZ max 20 |
| KNNK | **Excluded** | 736 non-draws in 12.5M — per-class fidelity undefined at this imbalance |

---

## Files

```
paper/
├── README.md           ← this file
├── intro.md            — motivation, oracle setting, contributions
├── setup.md            — Syzygy, endgames, feature families, fidelity definition
├── decomposition.md    — baseline → leakage exhibit → cursed enumeration → RF test
├── discussion.md       — method generality; leakage detection; open questions
└── figures/
    ├── fidelity_bar_{endgame}.png     — per-class fidelity (baseline)
    └── draw_boundary_errors.png       — error analysis plots
```

`[RESULT: ...]` placeholders in `.md` files = not yet confirmed.
Do not fill them until the RF test completes.

---

## Scope Boundaries

- No neural networks or learned representations
- No SAT/ILP rule composers
- No 7-piece or non-enumerable endgames
- No stochastic/MDP extensions
- No emergent-concept analysis (rule of the square, opposition, Lucena)
- No DTZ as a model feature (the leakage exhibit is why)
