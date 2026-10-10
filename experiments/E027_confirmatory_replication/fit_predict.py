#!/usr/bin/env python3
"""Create locked predictions without reading fresh test targets."""
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
POSTHOC = HERE.parent / "posthoc_decision_diagnostics"
TRACE_INDEX = np.arange(0, 401, 50)
VELOCITY_SIGMA = 0.01
FORCE_SIGMA = 0.04
FORCE_RHO = 0.8


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(label: str, probe: str) -> int:
    return int(hashlib.sha256(f"E027:{label}:{probe}".encode()).hexdigest()[:8], 16)


def noisy_trace(data: dict, probe: str, label: str, *, original_posthoc: bool = False) -> np.ndarray:
    trace = data[f"{probe}__trace"].astype(np.float32).copy()
    if original_posthoc:
        seed = int(hashlib.sha256(
            f"posthoc-sampling:{label}:{probe}".encode()).hexdigest()[:8], 16)
    else:
        seed = seed_for(label, probe)
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
    return trace


def original_observed(data: dict, probe: str, split: str) -> np.ndarray:
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


def flat(trace: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.concatenate((trace.reshape(len(trace), -1), goal[:, None]), axis=1)


def sampling_model(name: str, train_x: np.ndarray, train_y: np.ndarray):
    if name.startswith("ridge_"):
        return make_pipeline(StandardScaler(), Ridge(alpha=float(name.split("_", 1)[1]))).fit(
            train_x, train_y)
    if name.startswith("random_"):
        scaler = StandardScaler().fit(train_x)
        rng = np.random.default_rng(27027)
        projection = rng.normal(0, 1 / np.sqrt(train_x.shape[1]),
                                (train_x.shape[1], 256)).astype(np.float32)
        features = np.maximum(0, scaler.transform(train_x) @ projection)
        model = Ridge(alpha=float(name.split("_", 1)[1])).fit(features, train_y)
        return model, scaler, projection
    raise ValueError(name)


def sampling_predict(model, x: np.ndarray) -> np.ndarray:
    if isinstance(model, tuple):
        estimator, scaler, projection = model
        return estimator.predict(np.maximum(0, scaler.transform(x) @ projection))
    return model.predict(x)


def main() -> None:
    input_path = HERE / "data/test_inputs.npz"
    if not input_path.exists():
        raise RuntimeError("fresh test inputs are missing")
    fresh = load(input_path)
    calibration = load(SOURCE / "data/calibration.npz")
    validation = load(SOURCE / "data/validation.npz")
    selection = json.loads((SOURCE / "audit/selection.json").read_text())
    classical = joblib.load(SOURCE / "checkpoints/classical.joblib")
    predictions = {}

    # Frozen primary models from the registered matched-probe study.
    for probe in ("fixed_positive_first", "half_then_full_forward",
                  "fixed_negative_first", "multisine"):
        record = classical[probe]
        method = selection["probes"][probe]["primary_method"]
        trace = noisy_trace(fresh, probe, "primary_test")[:, TRACE_INDEX]
        x = flat(trace, fresh["goal"])
        if method == "random_feature":
            x = np.maximum(0, record["scaler"].transform(x) @ record["projection"])
        predictions[f"{probe}__primary"] = record["models"][method].predict(x).astype(np.float32)

    # Previously selected sampling-rate models, fitted on the original calibration split.
    sampling_choices = {
        "fixed_positive_first": {9: "random_1.0", 401: "ridge_10.0"},
        "multisine": {9: "ridge_100.0", 17: "ridge_0.1"},
    }
    for probe, choices in sampling_choices.items():
        train_full = noisy_trace(
            calibration, probe, "calibration", original_posthoc=True)
        test_full = noisy_trace(fresh, probe, "sampling_test")
        for count, name in choices.items():
            stride = 400 // (count - 1)
            indices = np.arange(0, 401, stride)
            train_x = flat(train_full[:, indices], calibration["goal"])
            test_x = flat(test_full[:, indices], fresh["goal"])
            model = sampling_model(name, train_x, calibration[f"{probe}__oracle_speed"])
            predictions[f"{probe}__samples_{count}"] = sampling_predict(model, test_x).astype(np.float32)

    # Locked structured half-then-full decoder.
    probe = "half_then_full_forward"
    train_x = flat(original_observed(calibration, probe, "calibration"), calibration["goal"])
    test_x = flat(noisy_trace(fresh, probe, "structured_test")[:, TRACE_INDEX], fresh["goal"])
    parameter_model = make_pipeline(StandardScaler(), Ridge(alpha=0.1)).fit(
        train_x, calibration["parameters"])
    estimated = parameter_model.predict(test_x)
    state_x = np.concatenate((calibration["parameters"], calibration["goal"][:, None]), axis=1)
    action_model = ExtraTreesRegressor(n_estimators=512, min_samples_leaf=8,
        max_features=1.0, n_jobs=-1, random_state=28128).fit(
            state_x, calibration[f"{probe}__oracle_speed"])
    predictions[f"{probe}__structured"] = action_model.predict(
        np.concatenate((estimated, fresh["goal"][:, None]), axis=1)).astype(np.float32)

    # Target parameters are intentionally inaccessible here. The evaluator fills this arm
    # after the prediction file is frozen, using the already-fitted action model checkpoint.
    checkpoint = HERE / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    joblib.dump(action_model, checkpoint / "half_then_full_action.joblib")
    output = HERE / "audit/predictions_before_targets.npz"
    np.savez_compressed(output, **predictions)
    record = {
        "status": "predictions_frozen_before_target_access",
        "test_input_sha256": sha256(input_path),
        "prediction_sha256": sha256(output),
        "test_targets_accessed": False,
        "methods": {
            "primary": {probe: selection["probes"][probe]["primary_method"]
                        for probe in ("fixed_positive_first", "half_then_full_forward",
                                      "fixed_negative_first", "multisine")},
            "sampling": sampling_choices,
            "structured": {"parameter": "ridge_0.1", "action": "tree_8"},
        },
    }
    (HERE / "audit/prediction_lock.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
