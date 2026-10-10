# E025 amendment 003 — feasible-set and budget implementation lock

Before any E025 world was generated, computational review found that evaluating
a dense 121-speed grid for every world and probe would duplicate the already
validated monotone release mapping at prohibitive cost. The successful release
set is therefore computed as one interval: 14 bisection iterations locate the
speeds whose stopping distances equal `goal - 0.06 m` and `goal + 0.06 m`.
Independent replay on the excluded smoke seed 25000 must verify monotonicity on
a 31-speed grid before primary generation. Any non-monotone world is reported,
excluded from interval certification, and never silently coerced.

The common absolute-work and absolute-path-displacement caps are the respective
maxima of the inherited positive-first probe over calibration worlds. Alternatives
are not allowed to raise either cap. D-optimal selection considers only probes
with no more cap violations than the inherited probe. The passive condition is
always retained as an information-floor control.

This amendment changes only the prespecified numerical construction of the same
successful-speed set and fixes the previously unspecified calibration cap
functional. Seeds, splits, probe waveforms, noise model, models, gates, and
decision rule are unchanged. No E025 source execution, primary data, model, or
result existed at this lock.
