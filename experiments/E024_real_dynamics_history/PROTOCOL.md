# E024 — real action-conditioned history under terrain shift

## Status

Prospective design registered before downloading or inspecting any primary
trajectory outcomes. The one TartanDrive 2.0 engineering run identified in
`datasets/tartandrive2/sample_manifest.json` is excluded from every split.
Modeling source and exact split manifests must be locked in an amendment before
primary data processing begins.

## Question

Does recent action-conditioned proprioceptive history improve short-horizon
vehicle-response prediction on held-out off-road conditions, beyond instantaneous
state, classical linear dynamics, and matched nonlinear controls?

This is a real-data anchor for the “evidence can matter” premise. It is not an
active-probing experiment because TartanDrive contains observational human-driven
trajectories and does not permit counterfactual intervention.

## Data and split constraints

- Source: TartanDrive 2.0 converted vector modalities, pinned upstream catalog
  commit `fe893b9b4132bfbd30ba84a839872b3a855e8176`.
- Exclude `2023-11-14-14-24-21_gupta` and every run lacking the required
  converted modalities.
- Primary held-out course: all runs labeled `slag_heap`.
- Secondary held-out condition: all runs labeled `damp`.
- Neither held-out group may contribute training or model-selection windows.
- Split remaining run groups—not windows—into training and validation using a
  deterministic hash rule fixed in the source-lock amendment.
- Required signals: actual pedal controls, steering angle, GPS odometry, IMU,
  wheel RPM, and shock position. Images are deferred; vector-only E024 cannot
  support a VLA or vision claim.

## Frozen preprocessing principles

Resample each modality to a common 10 Hz grid without crossing run boundaries.
Stable-sort timestamps, retain the final sample among exact duplicates, reject
non-finite rows, and never interpolate across gaps longer than 0.25 seconds.
Channel semantics, coordinate transforms, outlier thresholds, normalization,
history duration, prediction horizon, and minimum valid-window rules must be
specified in the source-lock amendment using training and excluded engineering
data only. No held-out target distribution may tune these rules.

## Required baselines

1. Constant-velocity/current-state predictor.
2. Instantaneous-state Ridge and ExtraTrees.
3. Linear ARX/system-identification model using the full permitted history.
4. Random-feature Ridge using the same history.
5. A compact temporal convolution or GRU selected on validation only.
6. Wrong-history control that pairs each current state with another training
   history while preserving the current-state and action marginals.

All models receive matched action and sensor budgets. Report parameter count,
training time, and inference time. No foundation-model baseline is justified
until vector history demonstrates reliable value.

## Outcomes and gates

Primary outcome: one-second body-frame vehicle-response prediction error,
reported separately for longitudinal velocity, lateral velocity, and yaw rate,
with a prespecified normalized aggregate. Also report calibration by terrain,
speed, action magnitude, wheel-slip proxy, and shock-response quartile.

The history-value gate requires, on the untouched `slag_heap` group:

- at least 10% lower aggregate error than the instantaneous-state reference;
- at least 5% lower error than linear ARX; and
- degradation of at least 5% under the wrong-history control.

All three paired run-block-bootstrap 95% intervals must exclude zero in the
registered direction. The `damp` group is a separately reported stress test and
cannot rescue a failed primary gate.

## Decision rule

If the gate passes, E024 supports the narrow premise that action-conditioned
history carries transferable real physical information; it still does not show
active probe value. If history fails to beat classical identification, do not
claim a learned-history contribution. If no history method beats instantaneous
state, reconsider the real-data task before adding vision or larger models.
