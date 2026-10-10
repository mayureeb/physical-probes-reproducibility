# Decision-directed probe study

## Question

Can an open-loop physical probe selected for downstream task completion outperform
a probe selected for physical-parameter information under the same action and
sensing budgets?

## Status and evidence boundary

This protocol and its source are locked before candidate-search data are generated.
Candidate selection uses only the design split. The selected waveforms are written
to a second immutable lock before fresh training, validation, or test worlds are
generated. Test physical parameters, successful-action intervals, and physical
outcomes remain sealed until every trace-conditioned test action is frozen.

The study tests one controlled delayed-resistance simulator. It does not establish
performance on hardware, autonomous vehicles, robot manipulation, or visual inputs.

## Physical system and budgets

- Simulator and parameter ranges: inherited without modification from the matched
  delayed-resistance launch task.
- Probe duration: 401 simulation steps at 0.002 seconds per step.
- Active portion: 301 steps followed by 100 rest steps.
- Maximum target speed: 1.2 m/s.
- Observations: velocity and fixture force at nine fixed indices
  `[0, 50, 100, 150, 200, 250, 300, 350, 400]`.
- Observation noise: velocity standard deviation 0.01; force standard deviation
  0.04 with AR(1) correlation 0.8 on the nine sampled observations.
- Downstream action: one release-speed command.
- Physical completion: final landing error at most 0.06 m.
- Path and work caps: the maxima attained by the positive-first reference probe on
  the design worlds. Only candidates with no design-world violation are eligible.
  Violations are also reported on every fresh split.

## Candidate library

The deterministic library contains the inherited positive-first and negative-first
sweeps, the inherited multisine and half-then-full probes, eight chirps, eight
multisines with locked phases, and sixteen piecewise waveforms with locked seeds.
Every waveform starts and ends at zero, obeys the target-speed cap, and has the same
duration and rest period.

## Candidate selection

The design split contains 512 worlds from seed 30,030 with the same balanced
32-value goal grid used in the matched-probe study.

The decision-directed score is five-fold out-of-fold physical completion. In each
fold, an ExtraTrees regressor maps the nine noisy velocity/force observations and
goal to the midpoint of the successful release-speed interval. The selected probe
maximizes out-of-fold completion, breaking ties by release-speed MAE and then by
probe name.

The parameter-information score is the log determinant of the local standardized
linear sensitivity Gram matrix, computed exactly as in the earlier D-optimal
control. The eligible probe with the largest value is selected. The two selections
are made independently from the same candidate library and design worlds.

## Fresh confirmatory splits

- Training: seed 30,130, 4,096 worlds.
- Validation: seed 30,230, 1,024 worlds.
- Test: seed 30,330, 2,048 worlds.

The confirmatory comparison contains the decision-directed selection, the
D-optimal selection, and both inherited signed fixed sweeps. Duplicate selections
are retained as aliases but simulated and fitted only once.

## Decoders

The primary decoder is ExtraTrees midpoint regression with 512 trees. Minimum leaf
size is selected from `{2, 4, 8, 16}` by validation completion, then speed MAE,
separately for each unique probe. The final model is fit to the combined training
and validation split.

A diagnostic structured decoder estimates the eight simulator parameters using
standardized Ridge regression. Its regularization is selected from
`{0.1, 1, 10, 100}` by validation parameter NMSE. A fixed numerical planner then
maps the estimated state and goal to a release speed. The same planner is also run
with true parameters as an evaluator-only ceiling.

## Registered comparisons

All intervals use a 20,000-resample paired world bootstrap. Exact paired sign tests
are reported and Holm-adjusted across the directional comparisons.

1. The decision-directed probe must exceed the D-optimal probe by at least five
   percentage points in primary physical completion, with a 95% interval excluding
   zero.
2. It must exceed the better signed fixed sweep by at least three percentage points,
   with a 95% interval excluding zero.
3. Its structured-planner completion must exceed the D-optimal structured planner,
   with a 95% interval excluding zero.
4. Replacing its estimated state with the true state must improve completion, with
   a 95% interval excluding zero. This is a positive-control localization test, not
   evidence for the probe-selection hypothesis.

Failure of comparisons 1 or 2 rejects the claim that decision-directed selection
improves on the strong matched controls in this setting. No result will be described
as evidence for an adaptive or closed-loop probing policy.

