#!/usr/bin/env python3
"""Run post hoc tolerance, finite-pool Bayes, and sampling-rate diagnostics."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
PROBES = ("passive", "fixed_positive_first", "half_then_full_forward",
          "fixed_negative_first", "multisine", "random_piecewise")
SAMPLING_PROBES = ("fixed_positive_first", "multisine")
SAMPLE_COUNTS = (9, 17, 41, 81, 401)
TOLERANCES_M = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20)
TEMPERATURES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
TRACE_INDEX = np.arange(0, 401, 50)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
VELOCITY_SIGMA = 0.01
FORCE_SIGMA = 0.04
FORCE_RHO = 0.8


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completion(prediction: np.ndarray, data: dict, probe: str) -> float:
    low = data[f"{probe}__interval_low"]
    high = data[f"{probe}__interval_high"]
    return float(np.mean((prediction >= low) & (prediction <= high)))


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


def covariance_inverse() -> tuple[np.ndarray, np.ndarray]:
    length = len(TRACE_INDEX)
    force = FORCE_SIGMA ** 2 * FORCE_RHO ** np.abs(
        np.arange(length)[:, None] - np.arange(length)[None, :])
    return np.eye(length) / VELOCITY_SIGMA ** 2, np.linalg.inv(force)


def posterior_action(observation: np.ndarray, goal: float, reference: dict,
                     probe: str, temperature: float) -> float:
    mask = np.isclose(reference["goal"], goal, atol=1e-7, rtol=0)
    means = reference[f"{probe}__trace"][mask][:, TRACE_INDEX].astype(float)
    low = reference[f"{probe}__interval_low"][mask].astype(float)
    high = reference[f"{probe}__interval_high"][mask].astype(float)
    if not len(means):
        raise RuntimeError(f"No same-goal reference worlds for goal {goal}")
    velocity_inv, force_inv = covariance_inverse()
    dv = means[:, :, 0] - observation[:, 0]
    df = means[:, :, 1] - observation[:, 1]
    squared = np.einsum("ni,ij,nj->n", dv, velocity_inv, dv)
    squared += np.einsum("ni,ij,nj->n", df, force_inv, df)
    log_weight = -0.5 * squared / temperature
    log_weight -= logsumexp(log_weight)
    weight = np.exp(log_weight)
    endpoints = np.unique(np.concatenate((low, high)))
    candidates = np.unique(np.concatenate(((low + high) / 2,
        (endpoints[:-1] + endpoints[1:]) / 2)))
    mass = np.array([weight[(low <= value) & (value <= high)].sum()
                     for value in candidates])
    best = np.flatnonzero(np.isclose(mass, mass.max(), atol=1e-12, rtol=0))
    return float(np.median(candidates[best]))


def pool_bayes(calibration: dict, validation: dict, test_inputs: dict,
               test_targets: dict) -> tuple[dict, dict[str, np.ndarray]]:
    result, predictions = {}, {}
    final_reference = {key: np.concatenate((calibration[key], validation[key]))
                       for key in calibration if key in validation}
    for probe in PROBES:
        validation_observed = observed(validation, probe, "validation")
        validation_predictions = {}
        for temperature in TEMPERATURES:
            values = np.array([posterior_action(obs, goal, calibration, probe, temperature)
                               for obs, goal in zip(validation_observed, validation["goal"])])
            validation_predictions[temperature] = values
        validation_scores = {temperature: completion(values, validation, probe)
                             for temperature, values in validation_predictions.items()}
        selected_temperature = min(TEMPERATURES,
            key=lambda value: (-validation_scores[value], value))
        test_observed = observed(test_inputs, probe, "test")
        fixed = np.array([posterior_action(obs, goal, final_reference, probe, 1.0)
                          for obs, goal in zip(test_observed, test_inputs["goal"])], dtype=np.float32)
        calibrated = np.array([posterior_action(obs, goal, final_reference, probe,
                              selected_temperature)
                               for obs, goal in zip(test_observed, test_inputs["goal"])], dtype=np.float32)
        predictions[f"{probe}__pool_bayes_fixed"] = fixed
        predictions[f"{probe}__pool_bayes_calibrated"] = calibrated
        result[probe] = {
            "validation_completion_by_temperature": {
                str(key): value for key, value in validation_scores.items()},
            "selected_temperature": selected_temperature,
            "test_fixed_completion": completion(fixed, test_targets, probe),
            "test_calibrated_completion": completion(calibrated, test_targets, probe),
            "test_fixed_speed_mae": float(np.mean(np.abs(
                fixed - test_targets[f"{probe}__oracle_speed"]))),
            "test_calibrated_speed_mae": float(np.mean(np.abs(
                calibrated - test_targets[f"{probe}__oracle_speed"])))
        }
    return result, predictions


def full_noise(data: dict, probe: str, split: str) -> np.ndarray:
    trace = data[f"{probe}__trace"].astype(np.float32).copy()
    seed = int(hashlib.sha256(
        f"posthoc-sampling:{split}:{probe}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    trace[:, :, 0] += rng.normal(0, VELOCITY_SIGMA,
                                 trace[:, :, 0].shape).astype(np.float32)
    base_rho = FORCE_RHO ** (1 / 50)
    innovation = rng.normal(0, FORCE_SIGMA,
                            trace[:, :, 1].shape).astype(np.float32)
    noise = np.empty_like(innovation)
    noise[:, 0] = innovation[:, 0]
    factor = np.sqrt(1 - base_rho ** 2)
    for index in range(1, noise.shape[1]):
        noise[:, index] = base_rho * noise[:, index - 1] + factor * innovation[:, index]
    trace[:, :, 1] += noise
    return trace


def flat(trace: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.concatenate((trace.reshape(len(trace), -1), goal[:, None]), axis=1)


def fit_sampling_models(train_x: np.ndarray, train_y: np.ndarray,
                        validation_x: np.ndarray, validation: dict,
                        probe: str) -> tuple[dict, dict]:
    trials, models = {}, {}
    for alpha in RIDGE_ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train_x, train_y)
        pred = model.predict(validation_x)
        trials[f"ridge_{alpha}"] = (completion(pred, validation, probe),
                                     float(np.mean(np.abs(pred - validation[f"{probe}__oracle_speed"]))))
        models[f"ridge_{alpha}"] = model
    for leaf in (2, 4, 8, 16):
        model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=leaf,
            max_features=1.0, n_jobs=-1, random_state=27027).fit(train_x, train_y)
        pred = model.predict(validation_x)
        trials[f"tree_{leaf}"] = (completion(pred, validation, probe),
                                   float(np.mean(np.abs(pred - validation[f"{probe}__oracle_speed"]))))
        models[f"tree_{leaf}"] = model
    scaler = StandardScaler().fit(train_x)
    train_standard = scaler.transform(train_x)
    validation_standard = scaler.transform(validation_x)
    rng = np.random.default_rng(27027)
    projection = rng.normal(0, 1 / np.sqrt(train_x.shape[1]),
                            (train_x.shape[1], 256)).astype(np.float32)
    train_random = np.maximum(0, train_standard @ projection)
    validation_random = np.maximum(0, validation_standard @ projection)
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(train_random, train_y)
        pred = model.predict(validation_random)
        trials[f"random_{alpha}"] = (completion(pred, validation, probe),
                                      float(np.mean(np.abs(pred - validation[f"{probe}__oracle_speed"]))))
        models[f"random_{alpha}"] = (model, scaler, projection)
    selected = min(trials, key=lambda key: (-trials[key][0], trials[key][1], key))
    return {"selected": selected, "trials": trials}, models


def sampling_sweep(calibration: dict, validation: dict, test_inputs: dict,
                   test_targets: dict) -> tuple[dict, dict[str, np.ndarray]]:
    result, predictions = {}, {}
    for probe in SAMPLING_PROBES:
        noisy = {
            "calibration": full_noise(calibration, probe, "calibration"),
            "validation": full_noise(validation, probe, "validation"),
            "test": full_noise(test_inputs, probe, "test"),
        }
        result[probe] = {}
        for count in SAMPLE_COUNTS:
            stride = 400 // (count - 1)
            indices = np.arange(0, 401, stride)
            if len(indices) != count or indices[-1] != 400:
                raise RuntimeError(f"Non-nested sampling grid for {count}")
            train_x = flat(noisy["calibration"][:, indices], calibration["goal"])
            validation_x = flat(noisy["validation"][:, indices], validation["goal"])
            test_x = flat(noisy["test"][:, indices], test_inputs["goal"])
            meta, models = fit_sampling_models(train_x,
                calibration[f"{probe}__oracle_speed"], validation_x, validation, probe)
            selected = meta["selected"]
            fitted = models[selected]
            if selected.startswith("random_"):
                model, scaler, projection = fitted
                test_prediction = model.predict(np.maximum(0, scaler.transform(test_x) @ projection))
            else:
                test_prediction = fitted.predict(test_x)
            test_prediction = np.asarray(test_prediction, dtype=np.float32)
            predictions[f"{probe}__samples_{count}"] = test_prediction
            result[probe][str(count)] = {
                "trace_indices": indices.tolist(),
                "selected_method": selected,
                "validation_completion": meta["trials"][selected][0],
                "validation_speed_mae": meta["trials"][selected][1],
                "test_completion": completion(test_prediction, test_targets, probe),
                "test_speed_mae": float(np.mean(np.abs(
                    test_prediction - test_targets[f"{probe}__oracle_speed"])))
            }
    return result, predictions


def tolerance_sweep(primary_predictions: dict, pool_predictions: dict,
                    test_inputs: dict) -> tuple[dict, dict[str, np.ndarray]]:
    try:
        sys.path.insert(0, str(SOURCE))
        import evaluate
    except ImportError as error:
        return {"status": "pending_compatible_mujoco_jax_runtime",
                "reason": str(error)}, {}
    selected = json.loads((SOURCE / "audit/selection.json").read_text())
    replay_predictions = {}
    for probe in PROBES:
        method = selected["probes"][probe]["primary_method"]
        replay_predictions[f"{probe}__primary"] = primary_predictions[f"{probe}__{method}"]
        replay_predictions[f"{probe}__pool_bayes_fixed"] = pool_predictions[
            f"{probe}__pool_bayes_fixed"]
        replay_predictions[f"{probe}__pool_bayes_calibrated"] = pool_predictions[
            f"{probe}__pool_bayes_calibrated"]
    physical = evaluate.physical_predictions(replay_predictions, selected)
    result, arrays = {}, {}
    goal = test_inputs["goal"]
    for label, values in physical.items():
        error = np.abs(values["distance"] - goal)
        arrays[f"{label}__landing_error"] = error.astype(np.float32)
        result[label] = {
            "mean_landing_error_m": float(error.mean()),
            "completion_by_tolerance_m": {
                str(tolerance): float(np.mean(error <= tolerance))
                for tolerance in TOLERANCES_M}
        }
    return result, arrays


def main() -> None:
    started = time.time()
    data = SOURCE / "data"
    audit = SOURCE / "audit"
    calibration = load(data / "calibration.npz")
    validation = load(data / "validation.npz")
    test_inputs = load(data / "test_inputs.npz")
    test_targets = load(audit / "test_targets.npz")
    primary_predictions = load(audit / "predictions_pre_evaluation.npz")
    if not np.array_equal(test_inputs["world_seed"], test_targets["world_seed"]):
        raise RuntimeError("Test target alignment failed")
    bayes_result, bayes_predictions = pool_bayes(
        calibration, validation, test_inputs, test_targets)
    sampling_result, sampling_predictions = sampling_sweep(
        calibration, validation, test_inputs, test_targets)
    tolerance_result, landing_arrays = tolerance_sweep(
        primary_predictions, bayes_predictions, test_inputs)
    source_paths = [data / "calibration.npz", data / "validation.npz",
                    data / "test_inputs.npz", audit / "test_targets.npz",
                    audit / "predictions_pre_evaluation.npz",
                    audit / "selection.json"]
    payload = {
        "status": "post_hoc_diagnostics_complete",
        "elapsed_seconds": time.time() - started,
        "source_sha256": {path.name: sha256(path) for path in source_paths},
        "test_target_alignment": True,
        "pool_bayes": bayes_result,
        "sampling_rate": sampling_result,
        "tolerance_sweep": tolerance_result,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(HERE / "predictions.npz",
                        **bayes_predictions, **sampling_predictions, **landing_arrays)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
