# Validation performed for release 1.0.0

Packaging checks were run on macOS ARM64, Python 3.12, using the dependency snapshots in `environments/`.

Passed:

- Parse every released Python file; check all pipeline script entrypoints.
- Check every copied source and reference file against its release hash. Original hashes are retained separately, and the one original test adjustment is documented.
- Regenerate all 36 candidate waveforms and match the four released selected waveforms exactly to the recorded arrays.
- Check reported discrete completion fractions against cohort sizes.
- Run original E024 preprocessing/wrong-history preflight tests and E025 observation/analysis checks.
- Run the original physical-completion predicate check with values strictly inside/outside its tolerance. The original check used decimal values exactly at the tolerance and failed because of floating-point representation; only test inputs were changed. The scientific predicate was not changed.
- Replay one frozen synthetic world with the original MJX probe and release implementation: goal 0.683871 m, landing 0.684105 m, absolute error 0.000234 m.
- Run the original E025 GRU's 80-epoch training and validation-checkpoint loop on a tiny synthetic dataset; decoded predictions are finite.
- Run a reduced E030 pipeline with two newly generated worlds per split in an isolated checkout: generation, decoder/state-estimator model selection, combined-split refitting, action freezing, planning, physical evaluation, and paired comparisons. The temporary smoke evaluator's hard-coded cohort size is reduced to two; the released scientific evaluator retains the original 2,048-world cohort. This does not rerun the 512-world probe search or reproduce its experimental outcomes.
- Recompute all E027–E029 descriptive completions and mean landing errors, paired 20,000-resample intervals, exact p-values, and Holm adjustments from the released physical-replay arrays; they match the recorded outputs within numerical comparison tolerance.
- Retrieve one public TartanDrive timestamp object and verify its bytes and SHA-256 against the original download manifest.

Not performed during packaging:

- Full A100 dataset generation/retraining/replay for every published study.
- Downloading every TartanDrive raw object or retraining all real-data models.
- Independent reproduction of historical E003–E022 pilot results.
- Validation of every CUDA/driver/operating-system combination.

The full original E023–E031 pipelines and post hoc analyses are included so readers can perform those reproductions. Full-run results must be compared with the immutable references rather than inferred from smoke-test success. Optional Warp backends are not required for the tested JAX/MJX path; their missing-import messages do not prevent the checks above.
