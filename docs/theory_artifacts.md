# Theory Artifacts

## Purpose

This document defines how **structures discovered during learning** are recorded,
classified, and interpreted in this project.

While the Feature Registry specifies the *coordinates* used to represent chess
endgame positions, this document concerns the *artifacts that emerge* when learning
algorithms operate over those coordinates.

Examples of theory artifacts include:

- Learned decision rules
- Symbolic invariants
- State-space partitions
- Compressed abstractions
- Discovered feature combinations

These artifacts are not assumed to be correct, complete, or human-interpretable.
They are treated as empirical outcomes of learning processes.


## Separation from the Feature Registry

It is essential to distinguish between:

- **Representation**: how game states are encoded  
- **Discovery**: what structure emerges from learning over that representation  

The Feature Registry is static and prior to learning.  
Theory artifacts are dynamic and produced *after* learning.

No assumptions about theory, strategy, or explainability are encoded in the
feature registry itself.


## Understanding Dimensions

Each theory artifact can be classified along three independent dimensions:

1. **Origin** — who discovered the artifact  
2. **Explainability to the engine** — whether the producing system can justify or
   internally represent the artifact  
3. **Explainability to humans** — whether the artifact admits a concise human-level
   explanation  

These dimensions are descriptive, not normative.


## Understanding Classes (0–7)

For convenience, the combinations of these dimensions may be summarized using
an **understanding class** code.

| Class | Discovered By | Explainable by Engine | Explainable by Human |
|------:|---------------|-----------------------|----------------------|
| 0 | Human | No | No |
| 1 | Human | No | Yes |
| 2 | Human | Yes | Yes |
| 3 | Engine | Yes | No |
| 4 | Engine | Yes | Yes |
| 5 | Engine | No | Yes |
| 6 | Engine | No | No |
| 7 | Joint | Yes | Yes |

These classes are **annotations**, not objectives.
They may change over time as understanding improves.

---

## Interpretation Notes

- An artifact classified as “not explainable by humans” may become explainable later.
- An artifact explainable by humans is not necessarily optimal or correct.
- Engine explainability refers to internal representations (e.g., symbolic rules,
  program traces, or compressible structures), not post-hoc probing.

Understanding classes describe the *state of knowledge at a given time*.


## Artifact Scope

This document applies to artifacts such as:

- Decision tree nodes and splits
- Symbolic expressions derived from models
- Learned invariants or constraints
- Discovered equivalence classes of positions
- Emergent features constructed post hoc

It does not apply to raw features or input representations.


## Usage Guidelines

- Theory artifacts should reference the feature IDs they depend on
- Understanding classes should be recorded when artifacts are logged
- Reclassification of artifacts is expected and encouraged
- This document may evolve independently of the feature registry


## Long-Term Goal

By maintaining a clear record of theory artifacts and their modes of understanding,
the project aims to study:

- How strategic structure emerges from geometry
- Which abstractions are necessary for perfect play
- How machine-discovered concepts relate to human theory
- When and why explainability breaks down

This document supports the broader objective of understanding
**the emergence of theory from first-principles representations**.
