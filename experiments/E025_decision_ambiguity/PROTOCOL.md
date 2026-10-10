# E025 — estimator-independent ambiguity at a physical decision boundary

## Status and motivation

Prospective protocol. Lock the protocol, probe library, simulator source,
pair-search source, seeds, and evaluator before generating E025 worlds. E023
showed that a feasible hidden-state oracle does not establish trace
identifiability: the best clean-trace reference completed only 39.84%. E024
showed real history is informative but improved only 1.99% over a strong
instantaneous nonlinear baseline. No E023 or E024 test world may be reused.

## Question

Under a fixed diagnostic-motion, sensing, safety, and time budget, can we
separate downstream failures caused by physically unavailable evidence from
failures caused by an estimator that did not use available evidence?

E025 tests a certificate and matched controls, not a new policy. A learned
diagnostic method is permitted only after this study reveals a residual gap that
fixed, random, and classical excitation do not close.

## Worlds and downstream decision

Use fresh dual-memory launch worlds from the E021-approved MJX mechanism with
new seeds: 25,025 for 4,096 calibration worlds, 25,125 for 1,024 validation
worlds, and 25,225 for 2,048 untouched test worlds. Retain the same launch,
release-speed limits, target distribution, stop simulation, and 0.06 m completion
tolerance. For every world, compute the full set of successful release speeds on
a fixed dense grid followed by boundary bisection. A pair is
**decision-conflicting** when its successful-speed sets are disjoint.

The evaluator-only hidden-state oracle must complete at least 99% of test worlds.
Worlds with no feasible successful release speed are reported and excluded from
pair certification, never silently reassigned.

## Matched probe library

Every active probe lasts 1.2 seconds, obeys the same 1.2 m/s speed limit, starts
and ends at rest, and is rejected if it exceeds the common displacement or
absolute-work cap established from the original 1.2 m/s fixed probe on calibration
worlds. The locked library contains:

1. passive/no extra probe;
2. the original forward fixed probe;
3. a half-speed then full-speed forward probe;
4. a symmetric forward/reverse triangular probe;
5. a bounded multisine probe with fixed phases and seed 25025;
6. a random piecewise-constant probe with seed 25125; and
7. a classical open-loop D-optimal choice from items 2–6, selected using only the
   calibration prior and a local linear sensitivity model.

If an action violates a cap in a world, that violation is a failure and is
reported. No probe receives extra observations, duration, amplitude, retries, or
world-specific hidden state. The passive condition receives the naturally
available pre-release trace over the same wall-clock interval.

## Observation model and ambiguity certificate

Downsample the causal velocity/fixture-force trace to 10 Hz before comparison.
Velocity is treated as observed without added noise, matching E023; force noise
is zero-mean Gaussian with sigma 0.04 and AR(1) correlation 0.8. A secondary
i.i.d. sigma-0.04 analysis is reported but cannot replace the primary result.
All covariance regularization is fixed from calibration only.

For each world and probe, search calibration and validation worlds for the
closest decision-conflicting observation distribution. Refine candidate pairs
by bounded parameter-space optimization that never uses test outcomes for model
selection. For equal-covariance Gaussian trace distributions, compute the exact
pairwise total variation from Mahalanobis separation. Le Cam's two-point bound
then gives an estimator-independent lower bound on expected decision failure for
an equally weighted conflicting pair. The per-world ambiguity score is the
largest certified lower bound among its retained conflicting neighbors.

Report neighbor-search recall against exhaustive search on a locked calibration
subset, optimizer convergence, pair parameters, trace distance, successful-speed
sets, and replay hashes. A numerical pair is accepted only if independent MJX
replay reproduces both traces and confirms disjoint successful-speed sets.

## Practical decoders and controls

Using calibration and validation data only, fit the same observation-to-release
speed decoders for each probe: linear Ridge, ExtraTrees, fixed random features,
and a three-seed compact GRU. Add a nearest-neighbor successful-set decoder as a
nonparametric control. Freeze preprocessing, hyperparameters, selected epochs,
and predictions before evaluator access. All decoders receive identical traces
and goals for a given probe.

Also evaluate shuffled histories, force-masked traces, velocity-masked traces,
and the hidden-state oracle. No VLA or vision baseline is justified in E025:
the question concerns information present in matched proprioceptive traces, and
adding pretrained visual representations would change the sensing budget.

## Primary outcomes and gates

For each probe report:

- certified ambiguity: mean score and fraction of test worlds with a certified
  failure lower bound of at least 10%;
- decision performance: physical completion and landing error for every decoder;
- action cost: duration, peak speed, displacement, absolute work, and cap failures;
- certificate utility: run-independent bootstrap association between ambiguity
  score and practical decoder failure; and
- available-versus-unused decomposition: observed decoder failure minus the
  certified lower-bound component, labeled as a gap rather than a causal count.

The certificate gate passes only if replay validates at least 99% of retained
witness pairs and ambiguity score predicts the validation-selected decoder's test
failure with Spearman correlation at least 0.40 whose 20,000-resample 95%
bootstrap interval excludes zero. The action-comparison gate requires a probe to
reduce the fraction of at-least-10%-ambiguous worlds by at least 50% relative to
the original fixed probe while meeting every matched budget.

## Decision rule

- If the certificate gate fails, do not build a learned diagnostic policy; audit
  the certificate or abandon the decomposition.
- If a fixed or D-optimal probe passes both gates and has no practical completion
  deficit, use that control and do not claim a need for learning.
- Only if the certificate is predictive and a residual task-conditioned gap
  remains after every matched control may E026 develop a learned selector.
- No E025 result alone establishes real-world active probing, VLA capability, or
  a paper contribution.
