#!/usr/bin/env python3
"""Select state estimators and freeze test estimates without target access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
BUDGETS = (9, 33, 101, 401)
PARAMETERS = ("mass", "mu", "damping", "gain", "strength", "tau1", "tau2", "mix")
VELOCITY_SIGMA = 0.01
FORCE_SIGMA = 0.04
STEP_RHO = 0.8 ** (1 / 50)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed(data: dict[str, np.ndarray], split: str) -> np.ndarray:
    trace = data["trace"].astype(np.float32).copy()
    seed = int(hashlib.sha256(f"E029:{split}:noise".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    trace[:, :, 0] += rng.normal(0, VELOCITY_SIGMA, trace[:, :, 0].shape).astype(np.float32)
    innovation = rng.normal(0, FORCE_SIGMA, trace[:, :, 1].shape).astype(np.float32)
    noise = np.empty_like(innovation)
    noise[:, 0] = innovation[:, 0]
    factor = np.sqrt(1 - STEP_RHO ** 2)
    for index in range(1, noise.shape[1]):
        noise[:, index] = STEP_RHO * noise[:, index - 1] + factor * innovation[:, index]
    trace[:, :, 1] += noise
    return trace


def features(trace: np.ndarray, budget: int) -> np.ndarray:
    index = np.rint(np.linspace(0, trace.shape[1] - 1, budget)).astype(int)
    return trace[:, index].reshape(len(trace), -1)


def wrap(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor,
                                      transformer=StandardScaler())


def candidates(seed: int) -> dict:
    result = {}
    for alpha in (0.1, 1.0, 10.0, 100.0):
        result[f"ridge_{alpha}"] = wrap(make_pipeline(StandardScaler(), Ridge(alpha=alpha)))
    for leaf in (2, 4, 8, 16):
        result[f"tree_{leaf}"] = wrap(ExtraTreesRegressor(
            n_estimators=384, min_samples_leaf=leaf, max_features=1.0,
            n_jobs=-1, random_state=seed))
    for width in (128, 256):
        for model_seed in (29029, 29030):
            result[f"mlp_{width}_seed{model_seed}"] = wrap(make_pipeline(
                StandardScaler(), MLPRegressor(hidden_layer_sizes=(width, width),
                    early_stopping=True, validation_fraction=0.1, max_iter=500,
                    n_iter_no_change=25, random_state=model_seed)))
    return result


def main() -> None:
    train = load(HERE / "data/train.npz")
    validation = load(HERE / "data/validation.npz")
    test = load(HERE / "data/test_inputs.npz")
    train_trace = observed(train, "train")
    validation_trace = observed(validation, "validation")
    test_trace = observed(test, "test")
    scale = train["parameters"].std(0) + 1e-8
    combined_y = np.concatenate((train["parameters"], validation["parameters"]))
    lower, upper = combined_y.min(0), combined_y.max(0)
    outputs, records = {}, {}
    checkpoints = HERE / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    for budget in BUDGETS:
        tx = features(train_trace, budget)
        vx = features(validation_trace, budget)
        qx = features(test_trace, budget)
        models = candidates(29029 + budget)
        scores = {}
        for name, model in models.items():
            model.fit(tx, train["parameters"])
            prediction = model.predict(vx)
            scores[name] = float(np.mean(((prediction - validation["parameters"]) / scale) ** 2))
        chosen = min(scores, key=lambda name: (scores[name], name))
        final_model = candidates(39029 + budget)[chosen]
        final_model.fit(np.concatenate((tx, vx)), combined_y)
        estimate = np.clip(final_model.predict(qx), lower, upper).astype(np.float32)
        outputs[f"budget_{budget}__estimated_parameters"] = estimate
        joblib.dump(final_model, checkpoints / f"budget_{budget}_estimator.joblib")
        records[str(budget)] = {"selected_model": chosen,
            "validation_parameter_nmse": scores,
            "estimate_ranges": {name: [float(estimate[:, i].min()),
                                        float(estimate[:, i].max())]
                                for i, name in enumerate(PARAMETERS)}}
        print(json.dumps({"budget": budget, "selected": chosen,
                          "validation_nmse": scores[chosen]}), flush=True)

    output_path = HERE / "audit/estimates_before_targets.npz"
    np.savez_compressed(output_path, **outputs)
    lock = {"status": "state_estimates_frozen_before_target_access",
            "test_input_sha256": sha256(HERE / "data/test_inputs.npz"),
            "estimate_artifact_sha256": sha256(output_path),
            "test_targets_accessed": False, "budgets": records}
    (HERE / "audit/estimation_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
