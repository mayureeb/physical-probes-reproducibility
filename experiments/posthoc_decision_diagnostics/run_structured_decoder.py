#!/usr/bin/env python3
"""Post hoc continuous parameter-to-decision decoder analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
PROBES = ("passive", "fixed_positive_first", "half_then_full_forward",
          "fixed_negative_first", "multisine", "random_piecewise")
TRACE_INDEX = np.arange(0, 401, 50)
VELOCITY_SIGMA = 0.01
FORCE_SIGMA = 0.04
FORCE_RHO = 0.8
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def seed_for(split: str, probe: str) -> int:
    return int(hashlib.sha256(f"E025:{split}:{probe}".encode()).hexdigest()[:8], 16)


def observed(data: dict, probe: str, split: str) -> np.ndarray:
    trace = data[f"{probe}__trace"][:, TRACE_INDEX].astype(np.float32).copy()
    rng = np.random.default_rng(seed_for(split, probe))
    trace[:, :, 0] += rng.normal(0, VELOCITY_SIGMA, trace[:, :, 0].shape)
    innovation = rng.normal(0, FORCE_SIGMA, trace[:, :, 1].shape).astype(np.float32)
    noise = np.empty_like(innovation)
    noise[:, 0] = innovation[:, 0]
    factor = np.sqrt(1 - FORCE_RHO ** 2)
    for index in range(1, noise.shape[1]):
        noise[:, index] = FORCE_RHO * noise[:, index - 1] + factor * innovation[:, index]
    trace[:, :, 1] += noise
    return trace


def trace_features(data: dict, probe: str, split: str) -> np.ndarray:
    trace = observed(data, probe, split)
    return np.concatenate((trace.reshape(len(trace), -1), data["goal"][:, None]), axis=1)


def state_features(parameters: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.concatenate((parameters, goal[:, None]), axis=1)


def completion(prediction: np.ndarray, data: dict, probe: str) -> float:
    return float(np.mean((prediction >= data[f"{probe}__interval_low"])
                         & (prediction <= data[f"{probe}__interval_high"])))


def fit_parameter_models(x: np.ndarray, parameters: np.ndarray) -> dict:
    models = {}
    for alpha in ALPHAS:
        models[f"ridge_{alpha}"] = make_pipeline(
            StandardScaler(), Ridge(alpha=alpha)).fit(x, parameters)
    for leaf in (2, 4, 8, 16):
        models[f"tree_{leaf}"] = ExtraTreesRegressor(
            n_estimators=512, min_samples_leaf=leaf, max_features=1.0,
            n_jobs=-1, random_state=28028).fit(x, parameters)
    return models


def fit_action_models(x: np.ndarray, target: np.ndarray) -> dict:
    models = {}
    for alpha in ALPHAS:
        models[f"ridge_{alpha}"] = make_pipeline(
            StandardScaler(), Ridge(alpha=alpha)).fit(x, target)
    for leaf in (2, 4, 8, 16):
        models[f"tree_{leaf}"] = ExtraTreesRegressor(
            n_estimators=512, min_samples_leaf=leaf, max_features=1.0,
            n_jobs=-1, random_state=28128).fit(x, target)
    return models


def main() -> None:
    calibration = load(SOURCE / "data/calibration.npz")
    validation = load(SOURCE / "data/validation.npz")
    test_inputs = load(SOURCE / "data/test_inputs.npz")
    test_targets = load(SOURCE / "audit/test_targets.npz")
    if not np.array_equal(test_inputs["world_seed"], test_targets["world_seed"]):
        raise RuntimeError("Test target alignment failed")
    results, predictions = {}, {}
    for probe in PROBES:
        train_trace = trace_features(calibration, probe, "calibration")
        validation_trace = trace_features(validation, probe, "validation")
        test_trace = trace_features(test_inputs, probe, "test")
        parameter_models = fit_parameter_models(train_trace, calibration["parameters"])
        action_models = fit_action_models(
            state_features(calibration["parameters"], calibration["goal"]),
            calibration[f"{probe}__oracle_speed"])
        trials = {}
        for parameter_name, parameter_model in parameter_models.items():
            estimated = parameter_model.predict(validation_trace)
            for action_name, action_model in action_models.items():
                prediction = action_model.predict(state_features(estimated, validation["goal"]))
                trials[f"{parameter_name}__{action_name}"] = {
                    "completion": completion(prediction, validation, probe),
                    "speed_mae": float(np.mean(np.abs(
                        prediction - validation[f"{probe}__oracle_speed"])))
                }
        selected = min(trials, key=lambda key: (
            -trials[key]["completion"], trials[key]["speed_mae"], key))
        parameter_name, action_name = selected.split("__", 1)
        parameter_model = parameter_models[parameter_name]
        action_model = action_models[action_name]
        estimated_test_parameters = parameter_model.predict(test_trace)
        structured = action_model.predict(state_features(
            estimated_test_parameters, test_inputs["goal"])).astype(np.float32)
        true_parameter = action_model.predict(state_features(
            test_targets["parameters"], test_inputs["goal"])).astype(np.float32)
        predictions[f"{probe}__structured"] = structured
        predictions[f"{probe}__true_parameter_action"] = true_parameter
        parameter_scale = calibration["parameters"].std(0) + 1e-8
        results[probe] = {
            "selected_parameter_model": parameter_name,
            "selected_action_model": action_name,
            "validation": trials[selected],
            "test_structured_completion": completion(structured, test_targets, probe),
            "test_structured_speed_mae": float(np.mean(np.abs(
                structured - test_targets[f"{probe}__oracle_speed"]))),
            "test_true_parameter_action_completion": completion(
                true_parameter, test_targets, probe),
            "test_true_parameter_action_speed_mae": float(np.mean(np.abs(
                true_parameter - test_targets[f"{probe}__oracle_speed"]))),
            "test_parameter_normalized_rmse": float(np.sqrt(np.mean(
                ((estimated_test_parameters - test_targets["parameters"])
                 / parameter_scale) ** 2))),
            "candidate_count": len(trials)
        }
    payload = {"status": "post_hoc_structured_decoder_complete",
               "test_target_alignment": True, "probes": results}
    (HERE / "structured_decoder_results.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(HERE / "structured_decoder_predictions.npz", **predictions)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
