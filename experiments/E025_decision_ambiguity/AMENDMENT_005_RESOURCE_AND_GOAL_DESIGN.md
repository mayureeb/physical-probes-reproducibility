# E025 amendment 005 — resource-bounded splits and same-goal witnesses

## Resource correction

The excluded smoke benchmark showed that the registered 4,096/1,024/2,048
split would require substantially more A100 time than the available experiment
budget because every probe needs paired feasible-set boundary simulations. Before
primary generation, the split is reduced to 512 calibration, 128 validation,
and 256 untouched test worlds. The 20,000-replicate bootstrap is unchanged.
This reduces precision; intervals and all 256 test outcomes must be reported.

## Goal conditioning correction

The downstream goal is known to the decision maker. With continuously unique
random goals, two otherwise similar worlds would be distinguishable merely from
their goal, so treating them as an ambiguous pair would be invalid. E025 therefore
uses 32 equally spaced goals from 0.3 m through 1.0 m, with balanced shuffled
assignment within every split. Physical parameters retain their original
continuous distributions and seeds. Candidate ambiguity witnesses are restricted
to worlds with the exact same goal.

These changes were made after excluded-seed timing and monotonicity checks only.
No primary world, model, validation result, or test result existed. Probes,
observation noise, downstream tolerance, baselines, certificate threshold, and
decision gates are unchanged.
