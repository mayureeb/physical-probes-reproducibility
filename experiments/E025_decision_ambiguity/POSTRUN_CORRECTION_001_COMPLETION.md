# E025 post-run correction 001 — completion predicate

The first E025 evaluation completed before this correction and is preserved as
`audit/results_before_completion_correction.json` with its original evaluation
log. Review of that output exposed an implementation inconsistency: every oracle
midpoint landed within approximately 0.00035 m of its goal, yet the evaluator
reported zero completion because it additionally required the launch actuator to
attain the requested command speed exactly.

That extra predicate was not part of the registered E025 completion definition.
The protocol inherits the E019–E023 physical-completion convention: terminal
landing error no greater than 0.06 m after the full stopping rollout. E020 and
E021 also record command attainment and stopping-horizon status as separate
diagnostics rather than conjoining them with landing completion.

The corrected evaluation therefore changes only the reported physical completion
predicate from

`landing_within_tolerance AND command_attained AND stopped_flag`

to

`landing_within_tolerance`.

Command-attainment and stopping fractions remain reported unchanged. The locked
data, test targets, predictions, selected models, ambiguity certificate,
bootstrap seed and replicates, and action-comparison gate are not regenerated or
changed. The corrected result is explicitly labeled and retains hashes of the
immutable evaluator inputs and the preserved first-pass output.

