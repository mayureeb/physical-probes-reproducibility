# E024 amendment 002 — zero-window implementation correction

The first preprocessing execution downloaded and aligned all primary runs but
stopped before producing split files, opening evaluator targets, or fitting any
model. One short training run (`meadows_2023-09-14-12-17-25`) yielded zero valid
windows under the locked 2.0-second history, 1.0-second horizon, and gap rule.
Its empty arrays had rank one, causing concatenation to fail.

The implementation now emits correctly shaped empty arrays for zero-window runs.
No data, split, signal, cleaning rule, window, model, metric, gate, or seed was
changed. The run remains in the training assignment and contributes zero windows.
This is a mechanical failure correction, recorded before any model fitting or
held-out outcome evaluation.
