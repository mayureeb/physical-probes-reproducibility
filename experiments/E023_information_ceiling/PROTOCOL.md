# E023 — fixed-probe information ceiling

## Status

Prospective scientific protocol. Lock this protocol and all E023 source before
generating E023 data or inspecting any E023 validation or test result. E022's
negative result motivates the question; no E022 test world is reused.

## Question and hypothesis

Does the fixed 1.2 m/s probe contain enough information for a high-capacity
predictor to choose a successful release, and how much of the remaining error
is caused by force-observation noise?

The positive-control hypothesis is that a validation-selected high-capacity
reference using the noise-free causal probe trace will complete at least 80% of
fresh dual-memory test worlds at the unchanged 0.06 m tolerance. This is an
information-ceiling diagnostic, not a claim that noise-free sensing is deployable.

## Fresh paired data

Use the E021-approved full probe-to-stop MJX backend and dual-memory worlds only.
Generate 4,096 training worlds (seed 23023), 1,024 validation worlds (23123),
and 1,024 untouched test worlds (23223). Retain the deterministic 401-step
velocity/fixture-force trace, goal, eight simulator parameters, 14-iteration
oracle release speed, and local landing sensitivity estimated at oracle speed
plus or minus 0.025 m/s.

From every clean trace, deterministically construct 16 independent force-noise
replicates with sigma 0.04 using split seed plus 900. Compare four paired views:
clean, the first noisy replicate, and the force average of 4 or 16 replicates.
Velocity is unchanged. Noise is generated from the recorded seed rather than
stored redundantly.

## High-capacity references and selection

Train three two-layer 128-unit GRU seeds (23, 46, 69) for every observation
view. The objective is sensitivity-weighted release-speed error; the weight is
clipped using training data only. Select each checkpoint by validation estimated
landing error. Also tune an ExtraTrees direct-speed regressor on the clean
training/validation split over a prespecified grid of leaf size {1, 2, 4} and
feature fraction {0.5, 1.0}; refit the selected configuration independently on
all four observation views.

Before loading test data, freeze preprocessing, selected epochs, the tree
configuration, and whether clean GRU-ensemble or clean ExtraTrees has the lower
validation estimated landing error. The selected clean family is the primary
evidence-conditioned reference. Test predictions are frozen before physical
evaluation.

## Gates and reporting

The information-sufficiency gate passes only if the validation-selected clean
reference completes at least 80% of test worlds and the evaluator-only oracle
completes at least 99%. Report all per-seed and ensemble results, ExtraTrees,
speed MAE and bias, landing error, target-speed quartiles, sensitivity quartiles,
and the paired clean/single/mean-4/mean-16 degradation.

If the clean gate passes, a later experiment may develop a deployable estimator.
If it fails, do not try another architecture on this test set: the next study
must change the sensing action or establish a tighter ambiguity bound. No result
here establishes VLA capability, real-world transfer, or a paper contribution.
