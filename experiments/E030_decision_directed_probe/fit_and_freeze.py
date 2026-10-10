#!/usr/bin/env python3
"""Fit registered decoders and freeze test outputs without target access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROBES = ("decision_directed", "d_optimal", "fixed_positive_first",
          "fixed_negative_first")
TRACE_INDEX = np.arange(0, 401, 50)
PARAMETERS = ("mass", "mu", "damping", "gain", "strength", "tau1", "tau2", "mix")


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed(data: dict[str, np.ndarray], probe: str, split: str) -> np.ndarray:
    value = data[f"{probe}__trace"][:, TRACE_INDEX].astype(np.float32).copy()
    seed = int(hashlib.sha256(f"E030:{split}:{probe}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    value[:, :, 0] += rng.normal(0, 0.01, value[:, :, 0].shape)
    innovation = rng.normal(0, 0.04, value[:, :, 1].shape).astype(np.float32)
    noise = np.empty_like(innovation)
    noise[:, 0] = innovation[:, 0]
    factor = np.sqrt(1 - 0.8 ** 2)
    for index in range(1, noise.shape[1]):
        noise[:, index] = 0.8 * noise[:, index - 1] + factor * innovation[:, index]
    value[:, :, 1] += noise
    return value


def features(trace: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.concatenate((trace.reshape(len(trace), -1), goal[:, None]), axis=1)


def parameter_model(alpha: float):
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        transformer=StandardScaler())


def main() -> None:
    train = load(HERE / "data/train.npz")
    validation = load(HERE / "data/validation.npz")
    test = load(HERE / "data/test_inputs.npz")
    outputs, records = {}, {}
    checkpoints = HERE / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    scale = train["parameters"].std(0) + 1e-8
    combined_parameters = np.concatenate((train["parameters"], validation["parameters"]))
    lower, upper = combined_parameters.min(0), combined_parameters.max(0)
    for probe in PROBES:
        train_trace = observed(train, probe, "train")
        validation_trace = observed(validation, probe, "validation")
        test_trace = observed(test, probe, "test")
        tx = features(train_trace, train["goal"])
        vx = features(validation_trace, validation["goal"])
        qx = features(test_trace, test["goal"])
        y = train[f"{probe}__oracle_speed"]
        validation_y = validation[f"{probe}__oracle_speed"]
        validation_low = validation[f"{probe}__interval_low"]
        validation_high = validation[f"{probe}__interval_high"]
        tree_scores = {}
        for leaf in (2, 4, 8, 16):
            model = ExtraTreesRegressor(n_estimators=512, min_samples_leaf=leaf,
                max_features=1.0, n_jobs=-1, random_state=30030).fit(tx, y)
            prediction = model.predict(vx)
            tree_scores[str(leaf)] = {"completion": float(np.mean(
                (prediction >= validation_low) & (prediction <= validation_high))),
                "speed_mae": float(np.mean(np.abs(prediction - validation_y)))}
        leaf = min((2, 4, 8, 16), key=lambda value: (
            -tree_scores[str(value)]["completion"], tree_scores[str(value)]["speed_mae"], value))
        direct = ExtraTreesRegressor(n_estimators=512, min_samples_leaf=leaf,
            max_features=1.0, n_jobs=-1, random_state=30130)
        direct.fit(np.concatenate((tx, vx)), np.concatenate((y, validation_y)))
        outputs[f"{probe}__direct"] = direct.predict(qx).astype(np.float32)
        joblib.dump(direct, checkpoints / f"{probe}_direct.joblib")

        parameter_scores = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            model = parameter_model(alpha).fit(tx, train["parameters"])
            estimate = model.predict(vx)
            parameter_scores[str(alpha)] = float(np.mean(
                ((estimate - validation["parameters"]) / scale) ** 2))
        alpha = min((0.1, 1.0, 10.0, 100.0),
                    key=lambda value: (parameter_scores[str(value)], value))
        estimator = parameter_model(alpha)
        estimator.fit(np.concatenate((tx, vx)), combined_parameters)
        estimate = np.clip(estimator.predict(qx), lower, upper).astype(np.float32)
        outputs[f"{probe}__estimated_parameters"] = estimate
        joblib.dump(estimator, checkpoints / f"{probe}_state.joblib")
        records[probe] = {"tree_leaf": leaf, "tree_validation": tree_scores,
                          "ridge_alpha": alpha,
                          "parameter_validation_nmse": parameter_scores,
                          "estimate_ranges": {name: [float(estimate[:, i].min()),
                                                       float(estimate[:, i].max())]
                                              for i, name in enumerate(PARAMETERS)}}
        print(json.dumps({"probe": probe, "leaf": leaf, "alpha": alpha}), flush=True)
    path = HERE / "audit/estimates_and_direct_actions.npz"
    np.savez_compressed(path, **outputs)
    lock = {"status": "estimates_and_direct_actions_frozen_before_test_targets",
            "test_input_sha256": sha256(HERE / "data/test_inputs.npz"),
            "artifact_sha256": sha256(path), "test_targets_accessed": False,
            "probes": records}
    (HERE / "audit/estimation_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()

