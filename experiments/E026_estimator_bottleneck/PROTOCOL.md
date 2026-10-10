# E026 fresh-seed estimator-bottleneck study

## Status

Registered after E025 evaluation and before E026 data generation. This is a
confirmatory follow-up for the estimator-bottleneck interpretation, not part of
the original E025 gate.

## Question

When an active probe nearly removes the E025 finite-pool ambiguity certificate,
does practical decision completion improve with more independent training
worlds or with a decision-aware objective?

## Design

- Fresh world seeds: 26026 (train, 4096), 26126 (validation, 512), and 26226
  (test, 512).
- Probes: the E025 positive-first fixed probe and D-optimal multisine probe.
- The simulator, goal grid, observation channel, trace downsampling, force-noise
  covariance, successful-action intervals, and physical completion tolerance
  are inherited unchanged from E025.
- Data-scaling sizes are 256, 512, 1024, 2048, and 4096, using nested prefixes
  of the fresh training split. For each size and probe, validation chooses among
  Ridge, ExtraTrees, and fixed random-feature regression. All choices are frozen
  before test targets are loaded.
- The decision-aware comparison uses all 4096 training worlds. A candidate-speed
  classifier is trained on whether each candidate lies in the successful-action
  interval. Validation selects tree complexity. It is compared with the best
  full-data midpoint regressor selected under the same validation rule.

## Outcomes

Primary outcomes are held-out command-in-interval completion and Wilson 95%
intervals. Secondary outcomes are speed MAE and probe ranking. The scaling study
supports a data-limited explanation only if completion rises monotonically and
the 4096-world point exceeds the 512-world point by at least 10 percentage
points. The objective study supports loss mismatch only if the decision-aware
method exceeds the selected midpoint regressor by at least 5 percentage points
with a paired bootstrap 95% interval above zero.

The test-target file is generated separately and is loaded only after all
validation selections have been written to `audit/selection.json`.
