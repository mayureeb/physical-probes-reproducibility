# Amendment 3: increase the generation batch size

The first confirmatory generation attempt completed one 64-world engineering
batch in 225.7 seconds. This batch size underfilled the A100 and projected a
multi-hour run. The process was stopped before any split was completed or saved.

The generation batch size is increased from 64 to 256 worlds. Split seeds,
world counts, candidate selections, physical simulation, successful-action
calculation, models, hypotheses, and statistical tests are unchanged. The
partial in-memory output from the stopped process is excluded and generation
restarts from the beginning.
