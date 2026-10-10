# Confirmatory replication of decision diagnostics

## Status

Prospective protocol. The hypotheses below were chosen from the completed post hoc diagnostic package. This protocol, source, seeds, input hashes, model choices, and analysis rules must be hashed before the fresh worlds are generated. The test targets remain unopened until predictions are written and hashed.

## Purpose

The study tests whether the main post hoc patterns repeat on a larger independent set. It does not reclassify the earlier analyses as prospective evidence.

## Locked data and models

- Training and validation data are the existing matched-probe calibration and validation splits.
- The fresh test split contains 4,096 worlds generated with seed 27,227.
- The evaluated probes are positive-first, negative-first, half-then-full, and multisine.
- Sensor noise follows the existing continuous-time-consistent process: velocity standard deviation 0.01, force standard deviation 0.04, and force correlation 0.8 over the original 9-observation spacing.
- Primary decoders and their preprocessing are loaded from the frozen matched-probe checkpoint.
- Sampling-rate models use the previously selected methods without new validation or test selection: positive-first uses random features with alpha 1.0 at 9 samples and Ridge with alpha 10.0 at 401 samples; multisine uses Ridge with alpha 100.0 at 9 samples and Ridge with alpha 0.1 at 17 samples.
- The structured half-then-full decoder uses Ridge with alpha 0.1 for parameter estimation and ExtraTrees with minimum leaf size 8 for the parameter-to-action map. The same action map receives either estimated or true parameters.

## Confirmatory hypotheses

All outcomes use physical replay, not interval membership alone.

1. Positive-first completion exceeds multisine completion at tolerances 0.10 and 0.20 m.
2. Negative-first completion exceeds multisine completion at tolerances 0.10 and 0.20 m.
3. For multisine, 17 observations exceed 9 observations at the registered 0.06 m tolerance.
4. For positive-first, 401 observations perform worse than 9 observations at 0.06 m.
5. For half-then-full, the action model supplied with true parameters exceeds the same action model supplied with parameters estimated from the trace at 0.06 m.

## Analysis

Each comparison is paired by physical world. Report the completion difference and a 20,000-replicate paired percentile-bootstrap 95% interval. Also report an exact two-sided test over discordant paired outcomes, with Holm correction across the seven locked comparisons. A hypothesis is confirmed only when the observed direction matches the protocol and its Holm-adjusted value is below 0.05. All other probe, tolerance, error, and parameter metrics are descriptive.

## Evidence boundary

The study confirms or rejects these particular patterns under the locked simulator, sensor model, and model families. It does not establish autonomous-vehicle, robot, visual, or sim-to-real performance.
