# E025 amendment 006 — finite-pool witness certificate

Before primary generation, the pair-search implementation was narrowed from
continuous parameter-space refinement to exhaustive search over the complete
same-goal calibration-plus-validation reference pool. Up to the 32 strongest
nonzero witnesses per probe are independently replayed from their stored world
seeds and parameters. Exhaustive finite-pool search has recall one within that
pool and avoids data-dependent optimizer choices on test worlds.

The resulting quantity remains an estimator-independent Le Cam lower bound for
each retained pair, but it is **not** a global identifiability bound over the
continuous world distribution. Reports must call it a finite-pool witness
certificate. Failure to find a witness is absence of evidence, not proof of
identifiability.

This conservative change reduces the strength of any positive claim and removes
an expensive, underspecified optimizer. No primary world, trace, model,
validation result, or test result existed. Noise, probes, matched budgets,
thresholds, bootstrap, and decision gates are unchanged.
