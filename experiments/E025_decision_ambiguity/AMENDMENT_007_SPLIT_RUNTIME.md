# E025 amendment 007 — split execution runtime

Preflight found that the persistent RunPod's validated MJX/JAX environment uses
Python 3.12, while the validated PyTorch/scikit-learn environment uses Python
3.10. Before primary generation, the evaluator was made independent of the
learning module by duplicating only locked scalar constants and probe names. The
pipeline now runs physical generation and evaluation with the MJX environment
and model fitting with the learning environment. Artifacts remain file-mediated,
and the evaluator still opens targets only after predictions are frozen.

No model, metric, seed, probe, data split, target, gate, or scientific semantics
changed. No primary E025 data or result existed when this operational correction
was made.
