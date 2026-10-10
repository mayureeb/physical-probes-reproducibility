# Physical probes: reproducibility repository

Code and recorded reference results for **Evaluating Physical Probes for Task Decisions under Hidden Dynamics**, by **Mayuree Binjolkar**, Independent Researcher.

This release provides executable simulation, training, model/probe selection, physical evaluation, statistical analysis, and TartanDrive preprocessing pipelines for the paper's primary E023–E031 studies and reported post hoc diagnostics. Raw TartanDrive data must be downloaded from its public upstream source; it is not redistributed here. See [the results map](docs/RESULTS.md) for exact coverage and historical exclusions.

## Set up

Use Python 3.12. The original workflow separated simulation and training environments; keeping them separate avoids incompatible NumPy/PyTorch requirements.

```sh
python3.12 -m venv .venv-sim
python3.12 -m venv .venv-ml
.venv-sim/bin/python -m pip install -r requirements-simulation.txt
.venv-ml/bin/python -m pip install -r requirements-ml.txt
```

Simulation requirements above provide the CPU backend. For a Linux NVIDIA GPU machine, install the matching CUDA backend:

```sh
.venv-sim/bin/python -m pip install 'jax[cuda12]==0.11.1'
```

The original simulation studies used an 80 GB NVIDIA A100. Full runs are expensive and require substantially more resources than the smoke checks. CPU replay is possible but full CPU experiments can be very slow. ML code selects CUDA when available; the released versions and their own package/device requirements must be satisfied. Validated macOS dependency snapshots are under `environments/`; these are validation environments, not claims about every dependency of the original GPU environment.

## Check the installation

```sh
.venv-ml/bin/python scripts/check_package.py
.venv-ml/bin/python scripts/smoke_training.py
.venv-sim/bin/python scripts/smoke_simulator.py
.venv-sim/bin/python scripts/smoke_pipeline.py --ml-python .venv-ml/bin/python
.venv-sim/bin/python scripts/reanalyze.py
```

The reduced E030 pipeline uses two worlds per split in a temporary checkout to check generation, selection/refitting, freezing, planning, and evaluation. Smoke outputs are not the paper's experimental results. Reanalysis recomputes published statistics from released landing errors and is distinct from generating fresh trajectories or retraining.

## Reproduce the experiments

```sh
.venv-ml/bin/python scripts/reproduce.py --study all \
  --simulation-python .venv-sim/bin/python \
  --ml-python .venv-ml/bin/python
```

For a single study, replace `all` with its identifier, for example `E030`. Prerequisites run automatically. `--dry-run` lists the steps without executing them. Use `--skip-dependencies` only if prerequisite artifacts already exist. Run in a fresh checkout: generation and evaluation write new artifacts under `experiments/`, while immutable recorded outputs remain in `reference_results/`. Generated artifacts and logs are ignored by Git. A failed step stops the pipeline; its log is in `run_logs/`.

The full runner executes the original seeded split sizes and training budgets. Source locks shipped with the code preserve the original provenance; they do not mean that a new reproduction was prospectively registered. Prediction and action locks created by the rerun enforce the original evaluation boundaries.

Afterward:

```sh
.venv-ml/bin/python scripts/compare_results.py
```

The comparison reports deviations and missing studies. The published results are reference measurements, not hard-coded outputs of the reproduction pipeline.

## Real-data access

`--study E024` downloads the six vector modalities and their timestamps from the public TartanDrive 2.0 object store, using `datasets/tartandrive2/catalog.json`. This catalog freezes the public run identifiers, terrain labels, and availability flags used in the original experiment. The original raw download hashes, window counts, feature order, held-out runs, and split assignments are in `reference_results/E024_real_dynamics_history/`.

The runner verifies every downloaded raw file against the recorded hashes before training. Do not silently replace missing runs or changed files. The upstream data and terms are at https://github.com/castacks/tartan_drive_2.0. Only vector signals are downloaded; no image or lidar data are required. The object-store `AUTH_...` segment is a public storage namespace, not a credential. No account token is supplied by these scripts. Upstream availability is outside this repository's control.

## Limits and validation

See [VALIDATION.md](VALIDATION.md) for checks actually performed. Full A100 retraining and every real-data download have not been repeated during packaging. A runnable reproduction recipe and the recorded outputs are provided; a bit-for-bit cross-platform guarantee is not made. The complete private research archive and unpublished work are excluded.

## Licensing and citation

Original code: [MIT](LICENSE). Author-created reference results, synthetic fixtures, protocols, and documentation: [CC BY 4.0](DATA_LICENSE.md). Third-party assets retain their original terms.

Please cite Mayuree Binjolkar, *Evaluating Physical Probes for Task Decisions under Hidden Dynamics*, PhysWorldAI workshop at NeurIPS 2026. A stable paper link will be added when available.
