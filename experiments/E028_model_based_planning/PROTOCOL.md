# Model-based planning after physical-state estimation

## Status

Prospective protocol. Lock this document, all source, seeds, model choices, and parent artifacts before generating the fresh test worlds. Test parameters remain hidden until trace-conditioned actions are predicted and hashed.

## Question

Does replacing a learned parameter-to-action map with numerical planning in the known simulator improve decisions when both methods receive the same trace-based physical-parameter estimate?

## Data

- Reuse the matched-probe calibration and validation splits for estimator fitting and selection.
- Generate 2,048 fresh test worlds with seed 28,228.
- Evaluate half-then-full and negative-first probes. The former gave the strongest structured-decoder result; the latter gave the highest completion in the confirmatory tolerance study.
- Use the existing nine-observation velocity and force channel with velocity standard deviation 0.01, force standard deviation 0.04, and force correlation 0.8 between adjacent observations.

## Parameter estimation

Estimate mass, friction, damping, controller gain, resistance strength, two memory time constants, and mixture weight from the probe trace without the downstream goal. Select among standardized multi-output Ridge models with alpha 0.1, 1, 10, or 100 and ExtraTrees models with minimum leaf size 2, 4, 8, or 16. Selection minimizes mean squared error after each parameter is standardized by its calibration standard deviation. Refit the selected family on calibration plus validation data. Clip test estimates to the combined reference range before planning.

## Decision methods

1. The frozen primary direct trace-to-action decoder.
2. A fixed ExtraTrees parameter-to-action map with 512 trees and minimum leaf size 8, trained on calibration plus validation physical parameters, goals, and oracle release speeds.
3. Numerical planning in the locked simulator using the same estimated parameters and known goal.
4. Evaluator-only numerical planning using the true test parameters.

For numerical planning, instantiate the estimated or true world, execute the selected probe, and solve for the release speed whose simulated stopping distance reaches the goal by 14 boundary-search steps. Physically replay every selected speed in the true world.

## Confirmatory outcomes

For each probe, report physical completion at 0.06 m and mean landing error. The two locked comparisons are:

1. estimated-parameter numerical planning versus the learned parameter-to-action map;
2. true-parameter numerical planning versus estimated-parameter numerical planning.

Report paired completion differences, 20,000-replicate paired percentile-bootstrap 95% intervals, and exact paired tests over discordant outcomes. Apply Holm correction across the four probe-comparison pairs. A directional result is confirmed only if the interval excludes zero in the locked direction and the adjusted value is below 0.05. Direct trace-to-action performance and parameter-level errors are descriptive.

## Evidence boundary

The simulator is assumed known to the numerical planner. This study separates estimation and planning errors within that setting; it does not show that the simulator can be learned from real data or that the method transfers to hardware.
