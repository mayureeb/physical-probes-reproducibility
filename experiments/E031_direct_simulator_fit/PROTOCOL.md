# Direct simulator fitting and downstream decisions

## Status

Prospective confirmatory protocol. Lock this document, source, seed, optimizer,
initialization, outcomes, and tests before generating fresh worlds. Test physical
parameters remain in a separate evaluator artifact until both action vectors are
saved and hashed.

## Question

Does direct maximum-likelihood fitting of the known simulator to a diagnostic
trace improve downstream completion relative to the inherited amortized state
estimator?

## Data and observation model

- Generate 2,048 fresh delayed-resistance worlds with seed 31,331.
- Use the fixed negative-first probe and the inherited goal grid and parameter
  ranges.
- Use the same nine evenly spaced velocity and force observations as the primary
  matched-probe study.
- Add independent velocity noise with standard deviation 0.01. Add force noise
  with standard deviation 0.04 and correlation 0.8 between adjacent retained
  observations. The noise stream is fixed by the string `E031:test:noise`.

## Estimators

The comparator is the already-frozen nine-observation estimator from the temporal
sensing study. It was trained before these worlds exist and supplies one estimate
of mass, friction, damping, actuator gain, resistance strength, two memory time
constants, and mixture weight.

The intervention directly minimizes the Gaussian negative log likelihood of the
nine observed velocity-force pairs under the known MuJoCo/MJX simulator. Velocity
residuals are scaled by 0.01. Force residuals are whitened using the registered
AR(1) covariance. Each world uses two deterministic starts: the inherited
estimate and one fixed-seed perturbation in bounded logit coordinates. Optimize
with forward-mode derivatives and Adam for 50 updates, initial rate 0.03 and
cosine decay. Select the start with the lower trace likelihood. All eight fitted
values remain inside the inherited generation ranges.

## Decision evaluation

Give each estimate and the known goal to the unchanged 14-step numerical planner.
Save and hash both action vectors before opening the evaluator parameters. Replay
both actions in the true world after the same probe state.

## Outcomes and test

The primary outcome is completion within 0.06 m. The registered contrast is
direct simulator fitting minus the inherited estimator. Report the paired
difference, a 20,000-replicate paired percentile-bootstrap 95% interval, and an
exact two-sided test over discordant outcomes. A directional improvement is
confirmed only if the interval is above zero and the exact p value is below 0.05.

Secondary outcomes are mean landing error, trace likelihood, aggregate parameter
NMSE, and parameter-wise RMSE. These are descriptive and do not replace the
primary result.

## Evidence boundary

The simulator family, parameter ranges, observation covariance, and planner are
known. The test asks whether per-world simulator fitting recovers useful state
from this observation channel. It does not address simulator misspecification,
hardware transfer, or visual sensing.

