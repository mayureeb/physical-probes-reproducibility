#!/usr/bin/env python3
"""Fit parameter estimators and freeze non-planner actions without target access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
PROBES = ("half_then_full_forward", "fixed_negative_first")
TRACE_INDEX = np.arange(0, 401, 50)
PARAMETERS = ("mass", "mu", "damping", "gain", "strength", "tau1", "tau2", "mix")
VELOCITY_SIGMA = 0.01
FORCE_SIGMA = 0.04
FORCE_RHO = 0.8


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_reference(data: dict, probe: str, split: str) -> np.ndarray:
    trace = data[f"{probe}__trace"][:, TRACE_INDEX].astype(np.float32).copy()
    seed = int(hashlib.sha256(f"E025:{split}:{probe}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    trace[:, :, 0] += rng.normal(0, VELOCITY_SIGMA, trace[:, :, 0].shape)
    innovation = rng.normal(0, FORCE_SIGMA, trace[:, :, 1].shape).astype(np.float32)
    noise = np.empty_like(innovation)
    noise[:, 0] = innovation[:, 0]
    factor = np.sqrt(1 - FORCE_RHO ** 2)
    for index in range(1, noise.shape[1]):
        noise[:, index] = FORCE_RHO * noise[:, index - 1] + factor * innovation[:, index]
    trace[:, :, 1] += noise
    return trace


def observed_test(data: dict, probe: str) -> np.ndarray:
    trace = data[f"{probe}__trace"].astype(np.float32).copy()
    seed = int(hashlib.sha256(f"E028:test:{probe}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    trace[:, :, 0] += rng.normal(0, VELOCITY_SIGMA, trace[:, :, 0].shape).astype(np.float32)
    base_rho = FORCE_RHO ** (1 / 50)
    innovation = rng.normal(0, FORCE_SIGMA, trace[:, :, 1].shape).astype(np.float32)
    noise = np.empty_like(innovation)
    noise[:, 0] = innovation[:, 0]
    factor = np.sqrt(1 - base_rho ** 2)
    for index in range(1, noise.shape[1]):
        noise[:, index] = base_rho * noise[:, index - 1] + factor * innovation[:, index]
    trace[:, :, 1] += noise
    return trace[:, TRACE_INDEX]


def flat(trace: np.ndarray, goal: np.ndarray | None = None) -> np.ndarray:
    values = trace.reshape(len(trace), -1)
    return values if goal is None else np.concatenate((values, goal[:, None]), axis=1)


def normalized_mse(prediction: np.ndarray, target: np.ndarray,
                   scale: np.ndarray) -> float:
    return float(np.mean(((prediction - target) / scale) ** 2))


def fit_candidates(x: np.ndarray, y: np.ndarray) -> dict:
    models = {}
    for alpha in (0.1, 1.0, 10.0, 100.0):
        models[f"ridge_{alpha}"] = make_pipeline(
            StandardScaler(), Ridge(alpha=alpha)).fit(x, y)
    for leaf in (2, 4, 8, 16):
        models[f"tree_{leaf}"] = ExtraTreesRegressor(
            n_estimators=512, min_samples_leaf=leaf, max_features=1.0,
            n_jobs=-1, random_state=28228).fit(x, y)
    return models


def refit(name: str, x: np.ndarray, y: np.ndarray):
    if name.startswith("ridge_"):
        return make_pipeline(StandardScaler(), Ridge(alpha=float(name.split("_", 1)[1]))).fit(x, y)
    return ExtraTreesRegressor(
        n_estimators=512, min_samples_leaf=int(name.split("_", 1)[1]),
        max_features=1.0, n_jobs=-1, random_state=28228).fit(x, y)


def main() -> None:
    test_input_path = HERE / "data/test_inputs.npz"
    if not test_input_path.exists():
        raise RuntimeError("test inputs are missing")
    calibration = load(SOURCE / "data/calibration.npz")
    validation = load(SOURCE / "data/validation.npz")
    test = load(test_input_path)
    classical = joblib.load(SOURCE / "checkpoints/classical.joblib")
    selection = json.loads((SOURCE / "audit/selection.json").read_text())
    parameter_scale = calibration["parameters"].std(0) + 1e-8
    reference_parameters = np.concatenate((calibration["parameters"], validation["parameters"]))
    lower, upper = reference_parameters.min(0), reference_parameters.max(0)
    outputs, records = {}, {}
    checkpoints = HERE / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    for probe in PROBES:
        calibration_trace = observed_reference(calibration, probe, "calibration")
        validation_trace = observed_reference(validation, probe, "validation")
        test_trace = observed_test(test, probe)
        train_x, validation_x, test_x = flat(calibration_trace), flat(validation_trace), flat(test_trace)
        candidates = fit_candidates(train_x, calibration["parameters"])
        scores = {name: normalized_mse(model.predict(validation_x),
                                       validation["parameters"], parameter_scale)
                  for name, model in candidates.items()}
        chosen = min(scores, key=lambda name: (scores[name], name))
        final_x = np.concatenate((train_x, validation_x))
        estimator = refit(chosen, final_x, reference_parameters)
        estimate = np.clip(estimator.predict(test_x), lower, upper).astype(np.float32)
        outputs[f"{probe}__estimated_parameters"] = estimate
        joblib.dump(estimator, checkpoints / f"{probe}_parameter_estimator.joblib")

        # Frozen direct decoder.
        direct_record = classical[probe]
        direct_method = selection["probes"][probe]["primary_method"]
        direct_x = flat(test_trace, test["goal"])
        if direct_method == "random_feature":
            direct_x = np.maximum(
                0, direct_record["scaler"].transform(direct_x) @ direct_record["projection"])
        outputs[f"{probe}__direct"] = direct_record["models"][direct_method].predict(
            direct_x).astype(np.float32)

        # Learned action map with the same parameter estimate used by the planner.
        action_x = np.concatenate((reference_parameters,
            np.concatenate((calibration["goal"], validation["goal"]))[:, None]), axis=1)
        action_y = np.concatenate((calibration[f"{probe}__oracle_speed"],
                                   validation[f"{probe}__oracle_speed"]))
        action_model = ExtraTreesRegressor(
            n_estimators=512, min_samples_leaf=8, max_features=1.0,
            n_jobs=-1, random_state=28328).fit(action_x, action_y)
        outputs[f"{probe}__learned_action"] = action_model.predict(
            np.concatenate((estimate, test["goal"][:, None]), axis=1)).astype(np.float32)
        joblib.dump(action_model, checkpoints / f"{probe}_action_model.joblib")
        records[probe] = {
            "validation_parameter_nmse": scores,
            "selected_parameter_model": chosen,
            "direct_method": direct_method,
            "test_estimate_range": {
                name: [float(estimate[:, index].min()), float(estimate[:, index].max())]
                for index, name in enumerate(PARAMETERS)},
        }

    estimate_path = HERE / "audit/estimates_and_actions_before_targets.npz"
    np.savez_compressed(estimate_path, **outputs)
    lock = {
        "status": "estimates_and_nonplanner_actions_frozen_before_target_access",
        "test_input_sha256": sha256(test_input_path),
        "artifact_sha256": sha256(estimate_path),
        "test_targets_accessed": False,
        "probes": records,
    }
    (HERE / "audit/estimation_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
