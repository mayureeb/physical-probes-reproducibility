#!/usr/bin/env python3
"""Freeze validation choices, then evaluate E026 scaling and decision loss."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROBES = ("fixed_positive_first", "multisine")
SIZES = (256, 512, 1024, 2048, 4096)
GRID = np.linspace(0.0, 3.0, 121, dtype=np.float32)
TRACE_INDEX = np.arange(0, 401, 50)
VELOCITY_SIGMA = 0.01
FORCE_SIGMA = 0.04
FORCE_RHO = 0.8


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed(data: dict, probe: str, split: str) -> np.ndarray:
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


def wilson(completed: int, count: int) -> list[float]:
    z = 1.959963984540054
    p = completed / count
    den = 1 + z * z / count
    center = (p + z * z / (2 * count)) / den
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / den
    return [float(center - half), float(center + half)]


def features(data: dict, probe: str, split: str) -> np.ndarray:
    return flat(observed(data, probe, split), data["goal"])


def metric(pred: np.ndarray, data: dict, probe: str) -> dict:
    low, high = data[f"{probe}__interval_low"], data[f"{probe}__interval_high"]
    good = (pred >= low) & (pred <= high)
    return {"completion": float(good.mean()), "completed": int(good.sum()),
            "wilson95": wilson(int(good.sum()), len(good)),
            "speed_mae": float(np.mean(np.abs(pred - data[f"{probe}__oracle_speed"])))}


def fit_regressors(x: np.ndarray, y: np.ndarray, vx: np.ndarray,
                   validation: dict, probe: str) -> tuple[dict, dict]:
    trials, models = {}, {}
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x, y)
        trials[f"ridge_{alpha}"] = metric(model.predict(vx), validation, probe)
        models[f"ridge_{alpha}"] = model
    for leaf in (2, 4, 8, 16):
        model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=leaf,
            max_features=1.0, n_jobs=-1, random_state=26026).fit(x, y)
        trials[f"tree_{leaf}"] = metric(model.predict(vx), validation, probe)
        models[f"tree_{leaf}"] = model
    rng = np.random.default_rng(26026)
    scaler = StandardScaler().fit(x)
    projection = rng.normal(0, 1 / np.sqrt(x.shape[1]), (x.shape[1], 256)).astype(np.float32)
    rx = np.maximum(0, scaler.transform(x) @ projection)
    rvx = np.maximum(0, scaler.transform(vx) @ projection)
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha).fit(rx, y)
        name = f"random_{alpha}"
        trials[name] = metric(model.predict(rvx), validation, probe)
        models[name] = (model, scaler, projection)
    chosen = min(trials, key=lambda name: (-trials[name]["completion"], trials[name]["speed_mae"]))
    return {"chosen": chosen, "trials": trials}, {chosen: models[chosen]}


def regression_predict(record: dict, model_record: dict, x: np.ndarray) -> np.ndarray:
    name = record["chosen"]
    model = model_record[name]
    if name.startswith("random_"):
        estimator, scaler, projection = model
        return estimator.predict(np.maximum(0, scaler.transform(x) @ projection))
    return model.predict(x)


def candidate_matrix(x: np.ndarray) -> np.ndarray:
    repeated = np.repeat(x, len(GRID), axis=0)
    speed = np.tile(GRID, len(x))[:, None]
    return np.concatenate((repeated, speed), axis=1)


def candidate_labels(data: dict, probe: str) -> np.ndarray:
    low = np.repeat(data[f"{probe}__interval_low"], len(GRID))
    high = np.repeat(data[f"{probe}__interval_high"], len(GRID))
    speed = np.tile(GRID, len(data["goal"]))
    return ((speed >= low) & (speed <= high)).astype(np.uint8)


def choose_candidates(model, x: np.ndarray) -> np.ndarray:
    probability = model.predict_proba(candidate_matrix(x))[:, 1].reshape(len(x), len(GRID))
    return GRID[np.argmax(probability, axis=1)]


def main() -> None:
    data_dir, audit = HERE / "data", HERE / "audit"
    train, validation, test_inputs = (load(data_dir / "train.npz"),
        load(data_dir / "validation.npz"), load(data_dir / "test_inputs.npz"))
    selection, fitted = {"status": "selected_without_test_targets", "probes": {}}, {}
    for probe in PROBES:
        tx, vx, qx = features(train, probe, "e026_train"), features(validation, probe, "e026_validation"), features(test_inputs, probe, "e026_test")
        fitted[probe] = {"scaling": {}, "decision": {}}
        probe_record = {"scaling": {}}
        for size in SIZES:
            record, model = fit_regressors(tx[:size], train[f"{probe}__oracle_speed"][:size], vx, validation, probe)
            probe_record["scaling"][str(size)] = record
            fitted[probe]["scaling"][str(size)] = model
        full_record = probe_record["scaling"][str(SIZES[-1])]
        classifiers, classifier_trials = {}, {}
        train_candidate = candidate_matrix(tx)
        labels = candidate_labels(train, probe)
        for leaf in (4, 8, 16, 32):
            model = ExtraTreesClassifier(n_estimators=192, min_samples_leaf=leaf,
                max_features=1.0, class_weight="balanced", n_jobs=-1,
                random_state=27027).fit(train_candidate, labels)
            pred = choose_candidates(model, vx)
            classifier_trials[str(leaf)] = metric(pred, validation, probe)
            classifiers[str(leaf)] = model
        chosen_leaf = min(classifier_trials, key=lambda value:
            (-classifier_trials[value]["completion"], classifier_trials[value]["speed_mae"]))
        probe_record["decision_objective"] = {"chosen_leaf": int(chosen_leaf),
            "trials": classifier_trials, "regression_reference": full_record["chosen"]}
        fitted[probe]["decision"] = classifiers[chosen_leaf]
        fitted[probe]["test_x"] = qx
        selection["probes"][probe] = probe_record
    audit.mkdir(parents=True, exist_ok=True)
    selection_path = audit / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")

    targets_path = audit / "test_targets.npz"
    targets = load(targets_path)
    if not np.array_equal(test_inputs["world_seed"], targets["world_seed"]):
        raise RuntimeError("test target alignment failed")
    test = {**test_inputs, **targets}
    result = {"status": "e026_evaluated", "selection_sha256": digest(selection_path),
              "targets_sha256": digest(targets_path), "probes": {}}
    rng = np.random.default_rng(2702701)
    for probe in PROBES:
        qx = fitted[probe]["test_x"]
        scaling = {}
        for size in SIZES:
            record = selection["probes"][probe]["scaling"][str(size)]
            pred = regression_predict(record, fitted[probe]["scaling"][str(size)], qx)
            scaling[str(size)] = {**metric(pred, test, probe), "method": record["chosen"]}
        decision_pred = choose_candidates(fitted[probe]["decision"], qx)
        full_record = selection["probes"][probe]["scaling"][str(SIZES[-1])]
        regression_pred = regression_predict(full_record, fitted[probe]["scaling"][str(SIZES[-1])], qx)
        diff = (((decision_pred >= test[f"{probe}__interval_low"]) &
                 (decision_pred <= test[f"{probe}__interval_high"])).astype(float) -
                ((regression_pred >= test[f"{probe}__interval_low"]) &
                 (regression_pred <= test[f"{probe}__interval_high"])).astype(float))
        boot = np.empty(20000)
        for index in range(len(boot)):
            selected = rng.integers(0, len(diff), len(diff))
            boot[index] = diff[selected].mean()
        objective = {"decision_aware": metric(decision_pred, test, probe),
                     "midpoint_regression": metric(regression_pred, test, probe),
                     "paired_difference": float(diff.mean()),
                     "paired_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist()}
        points = [scaling[str(size)]["completion"] for size in SIZES]
        result["probes"][probe] = {"scaling": scaling,
            "scaling_monotone": bool(np.all(np.diff(points) >= 0)),
            "scaling_4096_minus_512": float(points[-1] - points[1]),
            "decision_objective": objective}
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
