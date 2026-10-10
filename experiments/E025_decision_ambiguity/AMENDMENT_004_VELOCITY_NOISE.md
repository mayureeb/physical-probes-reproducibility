# E025 amendment 004 — nonsingular observation model

The protocol inherited E023's convention of adding noise only to force while
treating velocity as exact. That convention is unsuitable for an
estimator-independent total-variation certificate: continuous deterministic
velocity traces with any unequal sample have disjoint probability support,
making the formal ambiguity bound trivially zero even at differences far below
real sensor resolution.

Before any E025 data was generated, the primary observation model is corrected
to include independent zero-mean Gaussian velocity noise with sigma 0.01 m/s.
Force retains sigma 0.04 and AR(1) correlation 0.8. The registered force-only
model is retained as a secondary singular-limit sensitivity analysis, explicitly
labeled as such; it cannot replace the primary result.

This amendment makes the certificate mathematically nondegenerate and physically
interpretable. No world, trace, model, validation result, or test result existed
at the lock. Probes, actions, seeds, splits, downstream decisions, gates, and
decision rules are unchanged.
