# Discussion

## What This Paper Establishes

Three things are confirmed regardless of the RF test outcome:

1. **The draw-boundary pattern is real and concentrated.** Per-class recall
   drops sharply at DRAW; WIN and LOSS are near-perfect. This is not an artifact
   of ordinal label structure.

2. **The 50-move-rule hypothesis is falsified.** Zero cursed/blessed positions
   exist in KBNK (24.5M), KRK (399K), or KPK (331K). The clock is structurally
   absent from the error-producing mechanism in these endgames.

3. **The naive DTZ-as-feature test is a methodological trap.** Adding any
   oracle-derived quantity as a feature to predict another oracle-derived label
   produces circular leakage, not information-access measurement. The flat 1.000
   signal across structurally different endgames is the diagnostic signature.
   This failure mode is not specific to chess — it applies to any domain where
   candidate explanatory features and labels share a common oracle source.

## The Leakage Detection Method

The leakage signature generalises. An excluded-information test is circular if:

- The candidate excluded variable and the predicted label are both derived from
  the same ground-truth source (oracle, simulator, or verifier); AND
- Adding the variable as a feature produces implausibly high attributed share
  across domains with different causal structures.

Cross-endgame coherence is the diagnostic: if the share is uniformly high across
endgames where the hypothesised mechanism has different structural roles, the
result is leakage, not a finding. KRK draws arise from stalemate, not the
50-move rule — a valid excluded-information test would show a lower share for KRK
than for KBNK. The flat 1.000 across all three reveals the circular reference.

## What the RF Test Tells Us

A RandomForest (200 trees, matched CV) improves DRAW recall by +0.0064 while
simultaneously degrading LOSS recall by −0.0051. The coupled trade-off is
diagnostic: RF shifts its probability mass toward DRAW predictions broadly,
catching some true DRAW positions at the cost of LOSS precision. This is not
evidence that the draw boundary is learnable — a genuinely learnable boundary
would yield DRAW improvement without LOSS degradation.

Approximately 75% of draw-boundary errors are common to both classifiers.
These errors are intrinsic to the geometric feature representation: they cannot
be resolved by increasing model expressiveness while holding features fixed.
The natural follow-up is feature expansion — richer geometric encodings,
or a neural representation trained directly from board state.

## Connection to the Broader Research Programme

This paper is the prerequisite for symbolic rule extraction from tablebase-trained
models. The rule extraction programme asks: what did the model learn, and can
those learned patterns be surfaced as interpretable, human-readable rules?

Before extracting rules, you need to know which of the model's decisions are
trustworthy. A decision tree is a rule system — but its rules at the draw
boundary are wrong 2.6% of the time. Extracting rules indiscriminately would
produce incorrect rules for the boundary region.

The decomposition here establishes:
- WHERE the model is faithful (WIN and LOSS regions: >99.5%)
- WHERE it is not (DRAW boundary: 97.4%)
- WHAT causes the gap (geometric — not clock; learnable or intrinsic TBD)

Symbolic rule extraction should operate on the faithful region first. The
boundary region requires its own treatment — either a stronger feature set or an
explicit uncertainty annotation on extracted rules.

## Limitations

**Full enumeration as a luxury.** This method works because we can probe all
24.5M KBNK positions from the oracle. In real-world XAI settings, there is no
oracle and no full enumeration. The method's validity depends on oracle access.

**Static features only.** The geometric feature set is hand-engineered. A neural
encoding might resolve draw-boundary errors without explicit design — in which
case the errors were not intrinsic but representational. Testing this is left for
future work.

**Three endgames.** The cross-endgame replication uses three small endgames with
similar null-cursed results. Endgames where cursed positions genuinely exist
(certain 6- or 7-piece configurations) would provide the positive control the
current study lacks.

## Future Work

1. **Positive control endgame.** Find a tablebase endgame with a known non-zero
   cursed-position population and run the same pipeline. A valid excluded-
   information test should show non-zero cursed share in errors for that endgame.
   This would confirm the method produces true positives, not just true negatives.

2. **Feature expansion test.** Add richer geometric features (rule-of-the-square
   distances, opposition encodings, colour-corner interactions) to the draw-
   boundary error positions and retrain. If errors vanish: they were geometric
   excluded-information failures. If errors persist: they are intrinsic.

3. **Symbolic rule extraction (main branch).** Use the faithful WIN and LOSS
   regions to extract interpretable decision rules. Document the draw-boundary
   region as an explicit gap with known error characteristics.

## Conclusion

The 50-move-rule hypothesis is falsified by full enumeration: zero cursed or
blessed positions exist across KBNK (24.5M), KRK (399K), and KPK (331K).
All draw-boundary errors are geometric.

A matched-pair RF comparison rules out model-class underfitting as the primary
cause: 75% of draw-boundary errors persist under a stronger learner, and the
modest +0.64% DRAW improvement is coupled with −0.51% LOSS degradation,
indicating a calibration shift rather than boundary learning.

The draw-boundary geometry contains structural patterns that resist static
geometric representation. The correct next step is feature expansion — not
a stronger model class on the same features, but a richer view of the board
that might encode what the current feature set approximates imprecisely.
