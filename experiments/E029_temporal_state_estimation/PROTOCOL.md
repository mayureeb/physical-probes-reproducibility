# Temporal observations for physical state estimation

## Status

Prospective protocol. Lock this document, source, seeds, observation budgets,
candidate models, outcomes, and tests before generating any study data. Test
physical parameters remain hidden until all trace-conditioned actions are saved
and hashed.

## Question

Does denser temporal sensing improve physical-state estimation enough to improve
downstream decisions under a fixed diagnostic probe and numerical planner?

## Data

- Generate 4,096 training worlds with seed 29,029, 1,024 validation worlds with
  seed 29,129, and 2,048 test worlds with seed 29,229.
- Use delayed-resistance family 4 and the fixed negative-first probe, which gave
  the highest practical completion in the preceding planning study.
- Preserve the full 401-step velocity and force trace.
- Add velocity noise with standard deviation 0.01 and force noise with standard
  deviation 0.04. Force noise follows the same temporal process as the previous
  nine-observation channel, with correlation 0.8 across 50 simulator steps.
- Compare 9, 33, 101, and 401 evenly spaced observations from the same noisy
  trace. These conditions differ only in temporal sampling density.

## State estimators

Estimate mass, friction, damping, controller gain, resistance strength, two
memory time constants, and mixture weight. Standardize input features and score
validation predictions by mean squared error after standardizing each physical
parameter by its training standard deviation.

For every observation budget, validation selects among Ridge regression with
alpha 0.1, 1, 10, or 100; ExtraTrees with 384 trees and minimum leaf size 2, 4,
8, or 16; and multilayer perceptrons with hidden widths (128, 128) or (256, 256),
early stopping, and seeds 29,029 and 29,030. Refit the selected specification on
training plus validation worlds and clip estimates to the combined reference
range.

## Decision evaluation

For each observation budget, give the estimated physical parameters and known
goal to the locked 14-step numerical planner. Save and hash every selected action
before opening the test parameters. After the lock, compute an evaluator-only
true-state planner and physically replay all five action sets in the true worlds.

## Registered outcomes

The primary outcome is physical completion at 0.06 m. Mean landing error and
parameter-level normalized squared error are secondary outcomes. The locked
paired comparisons are:

1. 33 observations minus 9 observations;
2. 401 observations minus 9 observations;
3. true-state planning minus 401-observation estimated-state planning.

Use 20,000 paired bootstrap replicates for percentile 95% intervals and exact
paired tests over discordant completion outcomes. Apply Holm correction across
the three comparisons. A directional comparison is confirmed only when its
interval is above zero and its adjusted p value is below 0.05. The 101-observation
condition, estimator identities, parameter errors, and monotonicity across all
four budgets are descriptive.

## Evidence boundary

The simulator and probe are fixed and known to the planner. This study tests
temporal observation density within one synthetic delayed-resistance system. It
does not test learned simulator error, hardware transfer, or visual sensing.
