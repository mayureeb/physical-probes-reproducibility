# Amendment 001: inherited estimator clipping

## Timing

Recorded after fresh trace generation and after 256 of 2,048 fits had run, but
before any action was planned, saved, or evaluated and before the separate test
parameter artifact was opened.

## Issue

The first execution clipped the inherited estimator to the nominal simulator
ranges. The inherited temporal-sensing pipeline instead clipped predictions to
the elementwise range of its combined training and validation parameters. A
checkpoint replay check exposed this difference before target access.

## Correction

Apply the original combined training/validation range exactly, discard the
incomplete estimates from the interrupted process, and restart fitting for all
2,048 trace-only test inputs. The optimizer, starts, observations, seed, outcomes,
and tests are unchanged.
