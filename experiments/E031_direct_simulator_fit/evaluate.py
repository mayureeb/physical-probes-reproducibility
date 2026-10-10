#!/usr/bin/env python3
"""Open evaluator parameters after the action lock and replay both arms."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
E025 = HERE.parent / "E025_decision_ambiguity"
E028 = HERE.parent / "E028_model_based_planning"
sys.path.insert(0, str(E025))
sys.path.insert(0, str(E028))
import build_data as build  # noqa: E402
import plan_before_targets as planning  # noqa: E402

BATCH = 128
BOOTSTRAPS = 20_000
PARAMETERS = ("mass", "mu", "damping", "gain", "strength", "tau1", "tau2", "mix")


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(actions: dict[str, np.ndarray], parameters: np.ndarray,
           goals: np.ndarray, seeds: np.ndarray) -> dict[str, np.ndarray]:
    errors = {name: [] for name in actions}
    probe = build.jp.asarray(build.probes.library()["fixed_negative_first"])
    started = time.perf_counter()
    for start in range(0, len(goals), BATCH):
        end = min(start + BATCH, len(goals))
        worlds = planning.worlds(parameters[start:end], goals[start:end], seeds[start:end])
        _, models, initial = build.e020.make_models(worlds)
        params = build.e020.parameters(worlds)
        final, _ = build.batch_probe_with_cost(models, initial, params, probe)
        data, z1, z2, *_ = final
        for name, values in actions.items():
            outcome = build.e021.batch_release(models, data, params, z1, z2,
                build.jp.asarray(values[start:end]))
            jax.block_until_ready(outcome[0])
            errors[name].append(np.abs(np.asarray(outcome[0]) - goals[start:end]))
        print(json.dumps({"replayed": end, "count": len(goals),
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    return {name: np.concatenate(parts) for name, parts in errors.items()}


def paired(first: np.ndarray, second: np.ndarray) -> dict:
    delta = first.astype(np.int8) - second.astype(np.int8)
    rng = np.random.default_rng(3133199)
    samples = np.empty(BOOTSTRAPS, dtype=np.float32)
    for start in range(0, BOOTSTRAPS, 500):
        index = rng.integers(0, len(delta), size=(min(500, BOOTSTRAPS-start), len(delta)))
        samples[start:start + len(index)] = delta[index].mean(1)
    discordant = int(np.sum(delta != 0))
    first_only = int(np.sum(delta == 1))
    p_value = 1.0 if not discordant else float(binomtest(first_only, discordant).pvalue)
    return {"direct_completion": float(first.mean()),
        "inherited_completion": float(second.mean()),
        "difference": float(delta.mean()),
        "bootstrap_95_interval": np.quantile(samples, (0.025, 0.975)).tolist(),
        "discordant_worlds": discordant, "direct_only": first_only,
        "exact_two_sided_p": p_value}


def main() -> None:
    action_path = HERE / "audit/actions_before_targets.npz"
    lock = json.loads((HERE / "audit/action_lock.json").read_text())
    if sha256(action_path) != lock["action_artifact_sha256"]:
        raise RuntimeError("action artifact hash mismatch")
    actions = load(action_path)
    estimates = load(HERE / "audit/estimates_before_targets.npz")
    inputs = load(HERE / "data/test_inputs.npz")
    targets = load(HERE / "audit/test_targets.npz")
    if not np.array_equal(inputs["world_seed"], targets["world_seed"]):
        raise RuntimeError("test alignment failed")
    errors = replay(actions, targets["parameters"], inputs["goal"], inputs["world_seed"])
    complete = {name: values <= 0.06 for name, values in errors.items()}
    contrast = paired(complete["direct"], complete["inherited"])
    scale = targets["parameters"].std(0) + 1e-8
    state = {}
    for name in ("inherited", "direct"):
        difference = estimates[name] - targets["parameters"]
        state[name] = {"parameter_nmse": float(np.mean((difference / scale) ** 2)),
            "per_parameter_rmse": {parameter: float(np.sqrt(np.mean(difference[:, i] ** 2)))
                for i, parameter in enumerate(PARAMETERS)}}
    result = {"status": "direct_simulator_fit_evaluated", "worlds": len(inputs["goal"]),
        "primary": contrast,
        "directional_improvement_confirmed": bool(contrast["difference"] > 0
            and contrast["bootstrap_95_interval"][0] > 0
            and contrast["exact_two_sided_p"] < 0.05),
        "descriptive": {name: {"completion": float(complete[name].mean()),
            "mean_landing_error_m": float(errors[name].mean())} for name in errors},
        "state_estimation": state,
        "action_artifact_sha256": sha256(action_path),
        "target_sha256": sha256(HERE / "audit/test_targets.npz")}
    np.savez_compressed(HERE / "audit/physical_replay.npz",
                        **{f"{name}_landing_error": value for name, value in errors.items()})
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

