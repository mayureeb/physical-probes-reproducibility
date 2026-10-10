#!/usr/bin/env python3
"""Prespecified secondary E025 covariance analysis; does not fit or select models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
PROBES = ("passive", "fixed_positive_first", "half_then_full_forward",
          "fixed_negative_first", "multisine", "random_piecewise")
PRIMARY = {"passive": "gru_ensemble", "fixed_positive_first": "ridge",
           "half_then_full_forward": "ridge", "fixed_negative_first": "random_feature",
           "multisine": "random_feature", "random_piecewise": "random_feature"}
TRACE_INDEX = np.arange(0, 401, 50)
BOOTSTRAP_SEED = 2502502
BOOTSTRAP_REPLICATES = 20_000


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def certificate(probe: str, reference: dict, target: dict, rho: float) -> np.ndarray:
    ref = reference[f"{probe}__trace"][:, TRACE_INDEX]
    test = target[f"{probe}__trace"][:, TRACE_INDEX]
    length = len(TRACE_INDEX)
    velocity_inverse = np.eye(length) / 0.01 ** 2
    covariance = 0.04 ** 2 * rho ** np.abs(
        np.arange(length)[:, None] - np.arange(length)[None, :])
    force_inverse = np.linalg.inv(covariance)
    scores = np.zeros(len(test))
    for index in range(len(test)):
        same_goal = reference["goal"] == target["goal"][index]
        conflict = ((reference[f"{probe}__interval_high"] <
                     target[f"{probe}__interval_low"][index]) |
                    (reference[f"{probe}__interval_low"] >
                     target[f"{probe}__interval_high"][index]))
        candidates = np.flatnonzero(same_goal & conflict)
        if not len(candidates):
            continue
        dv = ref[candidates, :, 0] - test[index, :, 0]
        df = ref[candidates, :, 1] - test[index, :, 1]
        squared = np.einsum("ni,ij,nj->n", dv, velocity_inverse, dv)
        squared += np.einsum("ni,ij,nj->n", df, force_inverse, df)
        scores[index] = ndtr(-np.sqrt(max(0.0, float(squared.min()))) / 2.0)
    return scores


def main() -> None:
    calibration = load(HERE / "data/calibration.npz")
    validation = load(HERE / "data/validation.npz")
    inputs = load(HERE / "data/test_inputs.npz")
    targets = load(HERE / "audit/test_targets.npz")
    predictions = load(HERE / "audit/predictions_pre_evaluation.npz")
    reference = {key: np.concatenate((calibration[key], validation[key]))
                 for key in calibration if key in validation}
    combined = {**inputs, **targets}
    failures = []
    for probe in PROBES:
        prediction = predictions[f"{probe}__{PRIMARY[probe]}"]
        failures.append(~((prediction >= targets[f"{probe}__interval_low"]) &
                          (prediction <= targets[f"{probe}__interval_high"])))
    failures = np.stack(failures, axis=1).astype(float)
    result = {"status": "prespecified_secondary_analysis_after_primary_evaluation",
              "bootstrap_seed": BOOTSTRAP_SEED,
              "bootstrap_replicates": BOOTSTRAP_REPLICATES,
              "covariance_conditions": {}}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for rho in (0.0, 0.4, 0.8):
        matrix = np.stack([certificate(probe, reference, combined, rho)
                           for probe in PROBES], axis=1)
        point = float(spearmanr(matrix.ravel(), failures.ravel()).statistic)
        bootstrap = np.empty(BOOTSTRAP_REPLICATES)
        for replicate in range(BOOTSTRAP_REPLICATES):
            chosen = rng.integers(0, len(inputs["goal"]), len(inputs["goal"]))
            bootstrap[replicate] = spearmanr(
                matrix[chosen].ravel(), failures[chosen].ravel()).statistic
        result["covariance_conditions"][f"rho_{rho:.1f}"] = {
            "spearman": point,
            "ci95": np.nanquantile(bootstrap, (0.025, 0.975)).tolist(),
            "per_probe_ambiguity_mean": {
                probe: float(matrix[:, index].mean())
                for index, probe in enumerate(PROBES)},
            "per_probe_fraction_at_least_0p10": {
                probe: float(np.mean(matrix[:, index] >= 0.10))
                for index, probe in enumerate(PROBES)}}
    output = HERE / "audit/noise_covariance_robustness.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
