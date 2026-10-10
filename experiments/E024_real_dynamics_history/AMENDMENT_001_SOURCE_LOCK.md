# E024 amendment 001 — source and split lock

## Timing and scope

This amendment locks the executable study before downloading or processing any
primary E024 trajectory. The sole engineering sample remains permanently
excluded. `audit/source_lock.json` records hashes and every exact run assignment.

## Signals and coordinates

The source arrays are used without clipping, winsorization, smoothing, or manual
run removal. Non-finite arrays invalidate a run. Odometry columns 7 and 8 are the
two reported linear-velocity components and column 12 is yaw rate; the prediction
target is those three values one second later. The two channels in
`controls/control.npy` are conservatively named `control_0` and `control_1`
because the converted-array metadata does not establish a more specific unit.
Steering, IMU, four wheel-encoder, and four shock-position channels are retained
in their stored units. No claim depends on interpreting a wheel spike as slip.

## Alignment and windows

Each stream is stable-sorted by timestamp and the final row of an exact duplicate
timestamp is retained. Linear interpolation is evaluated on a 10 Hz grid only
inside every stream's support and never across a bracketing gap over 0.25 s.
Windows contain 20 samples (2.0 s), targets are 10 samples (1.0 s) ahead, and
window starts advance by two samples. A window is rejected if any required row is
invalid. Normalization moments come from training windows only.

The current-state inputs are `vx`, `vy`, `yaw_rate`, `control_0`, `control_1`,
and steering. History adds the same channels plus six IMU, four wheel, and four
shock channels. All baselines therefore receive matched current actions and
sensors; only the permitted temporal representation differs.

## Exact split rule and models

All `slag_heap` runs form the primary test split. All remaining `damp` runs form
the secondary test split. For other eligible runs, the first eight hex digits of
SHA-256(run name), interpreted as an integer, assign validation when modulo five
equals zero and training otherwise. The resulting exact assignments are in the
source lock: 39 train, 10 validation, 3 slag-heap test, and 2 damp test runs.

Validation selects Ridge alpha from {0.1, 1, 10, 100}, ExtraTrees leaf size from
{4, 16, 64}, and the better of a 512-unit fixed random-ReLU feature model and a
two-layer 96-unit GRU ensemble. ExtraTrees uses seed 24024; the projection uses
seed 24024; GRUs use seeds 24, 48, and 72 for 60 epochs. Test predictions are
written before evaluator targets are opened. The wrong-history control circularly
reassigns the first 19 rows within each run while preserving the current row.

## Frozen evaluation

The aggregate is RMSE after per-target scaling by training-set standard
deviation. The primary gate and directional run-block bootstrap are exactly those
in `PROTOCOL.md`, with 20,000 replicates and seed 2402401. The damp split cannot
rescue a failed slag-heap gate. Quartile audits cover speed, action magnitude,
wheel-channel disagreement, and shock response. E024 remains observational and
cannot support a counterfactual active-probing claim.
