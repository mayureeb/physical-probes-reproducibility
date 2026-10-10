#!/usr/bin/env python3
"""Evaluate E024 once from frozen predictions and sealed held-out targets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BOOTSTRAP_SEED = 2402401
BOOTSTRAP_REPLICATES = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def error(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((prediction - target) / scale) ** 2)))


def metrics(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray) -> dict:
    residual = prediction - target
    return {
        "aggregate_nrmse": error(prediction, target, scale),
        "channel_rmse": np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
        "channel_mae": np.mean(np.abs(residual), axis=0).tolist(),
        "windows": int(len(target)),
    }


def improvement(reference: np.ndarray, candidate: np.ndarray, target: np.ndarray,
                scale: np.ndarray) -> float:
    return 1.0 - error(candidate, target, scale) / error(reference, target, scale)


def block_bootstrap(reference: np.ndarray, candidate: np.ndarray, target: np.ndarray,
                    runs: np.ndarray, scale: np.ndarray) -> dict:
    unique = np.unique(runs)
    blocks = [np.flatnonzero(runs == run) for run in unique]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in chosen])
        values[replicate] = improvement(reference[indices], candidate[indices],
                                        target[indices], scale)
    point = improvement(reference, candidate, target, scale)
    low, high = np.quantile(values, [0.025, 0.975])
    return {"relative_error_reduction": point, "ci95": [float(low), float(high)],
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "run_blocks": int(len(unique))}


def binned_metrics(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray,
                   values: np.ndarray) -> list[dict]:
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    output = []
    for index in range(4):
        mask = (values >= edges[index]) & (values <= edges[index + 1]
                                            if index == 3 else values < edges[index + 1])
        output.append({"quartile": index + 1, "lower": float(edges[index]),
                       "upper": float(edges[index + 1]), "windows": int(mask.sum()),
                       "aggregate_nrmse": error(prediction[mask], target[mask], scale)})
    return output


def main() -> None:
    audit = HERE / "audit"
    selection_path = audit / "selection.json"
    predictions_path = audit / "predictions_pre_evaluation.npz"
    selection = json.loads(selection_path.read_text())
    predictions = load(predictions_path)
    preprocessing = load(HERE / "checkpoints" / "preprocessing.npz")
    scale = preprocessing["target_scale"]
    primary = str(selection["primary_history_method"])
    instant = str(selection["primary_instant_method"])
    result = {"status": "evaluated_once_after_prediction_freeze",
              "primary_history_method": primary, "instantaneous_reference": instant,
              "bootstrap": {"seed": BOOTSTRAP_SEED,
                            "replicates": BOOTSTRAP_REPLICATES}, "splits": {},
              "input_hashes": {"selection": sha256(selection_path),
                               "predictions": sha256(predictions_path)}}
    for split in ("test_slag_heap", "test_damp"):
        inputs_path = HERE / "data" / f"{split}_inputs.npz"
        target_path = audit / f"{split}_targets.npz"
        inputs, targets = load(inputs_path), load(target_path)
        if not np.array_equal(inputs["run"], targets["run"]) or not np.array_equal(
                inputs["time"], targets["time"]):
            raise RuntimeError(f"target alignment failed for {split}")
        available = {key.split("__", 1)[1]: value for key, value in predictions.items()
                     if key.startswith(f"{split}__")}
        split_result = {"methods": {name: metrics(value, targets["target"], scale)
                                    for name, value in available.items()},
                        "target_sha256": sha256(target_path),
                        "input_sha256": sha256(inputs_path)}
        comparisons = {
            "history_vs_instantaneous": block_bootstrap(
                available[instant], available[primary], targets["target"], targets["run"], scale),
            "history_vs_arx": block_bootstrap(
                available["arx"], available[primary], targets["target"], targets["run"], scale),
            "history_vs_wrong_history": block_bootstrap(
                available["primary_wrong_history"], available[primary], targets["target"],
                targets["run"], scale),
        }
        split_result["comparisons"] = comparisons
        current, history = inputs["current"], inputs["history"]
        calibration_values = {
            "speed": np.linalg.norm(current[:, :2], axis=1),
            "action_magnitude": np.linalg.norm(current[:, 3:6], axis=1),
            "wheel_slip_proxy": np.std(history[:, -1, 12:16], axis=1),
            "shock_response": np.linalg.norm(history[:, -1, 16:20], axis=1),
        }
        split_result["primary_calibration_quartiles"] = {
            name: binned_metrics(available[primary], targets["target"], scale, values)
            for name, values in calibration_values.items()}
        result["splits"][split] = split_result
    primary_comparisons = result["splits"]["test_slag_heap"]["comparisons"]
    thresholds = {"history_vs_instantaneous": 0.10, "history_vs_arx": 0.05,
                  "history_vs_wrong_history": 0.05}
    gates = {name: {"threshold": threshold,
                    "point_pass": primary_comparisons[name]["relative_error_reduction"] >= threshold,
                    "directional_ci_pass": primary_comparisons[name]["ci95"][0] > 0.0}
             for name, threshold in thresholds.items()}
    result["gate"] = {"components": gates,
                      "passed": all(item["point_pass"] and item["directional_ci_pass"]
                                    for item in gates.values())}
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
