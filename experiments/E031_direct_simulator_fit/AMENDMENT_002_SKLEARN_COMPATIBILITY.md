# Amendment 002: checkpoint compatibility check

## Timing

Recorded before restarting the corrected fit and before any E031 action planning,
target access, or evaluation.

## Check

The inherited checkpoint was written with scikit-learn 1.4.2 and loaded on the
RunPod with 1.6.1. Replaying the old E029 test inputs with the original clipping
produced a maximum absolute difference of 0.000005722 in the eight physical
quantities relative to the frozen E029 estimates. This is below the stored
float-level precision relevant to the planner. The checkpoint is retained rather
than refit, so E031 continues to compare against the previously selected model.
