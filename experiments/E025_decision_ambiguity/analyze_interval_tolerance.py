#!/usr/bin/env python3
"""Audit task tolerance against frozen E025 test predictions."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "audit"
TARGET_PATH = AUDIT / "test_targets.npz"
PREDICTION_PATH = AUDIT / "predictions_pre_evaluation.npz"
SELECTION_PATH = AUDIT / "selection.json"
OUTPUT_PATH = AUDIT / "interval_tolerance_analysis.json"
PROBES = ["passive", "fixed_positive_first", "half_then_full_forward",
          "fixed_negative_first", "multisine", "random_piecewise"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_completion(widths, mae):
    """Centered Gaussian reference calibrated by E|error| = MAE."""
    if mae == 0:
        return 1.0
    sigma = mae * np.sqrt(np.pi / 2.0)
    return float(np.mean(2.0 * ndtr(widths / (2.0 * sigma)) - 1.0))


targets = np.load(TARGET_PATH)
predictions = np.load(PREDICTION_PATH)
with SELECTION_PATH.open() as stream:
    selection = json.load(stream)

reference_probe = "fixed_positive_first"
reference_low = targets[f"{reference_probe}__interval_low"].astype(float)
reference_high = targets[f"{reference_probe}__interval_high"].astype(float)
reference_midpoint = (reference_low + reference_high) / 2.0
reference_widths = reference_high - reference_low
oracle = targets[f"{reference_probe}__oracle_speed"].astype(float)
if not np.allclose(oracle, reference_midpoint, atol=1e-10, rtol=0):
    raise RuntimeError("Frozen oracle speed is not the interval midpoint.")

probe_results = {}
for probe in PROBES:
    low = targets[f"{probe}__interval_low"].astype(float)
    high = targets[f"{probe}__interval_high"].astype(float)
    midpoint = (low + high) / 2.0
    method = selection["probes"][probe]["primary_method"]
    prediction = predictions[f"{probe}__{method}"].astype(float)
    mae = float(np.mean(np.abs(prediction - midpoint)))
    completion = float(np.mean((prediction >= low) & (prediction <= high)))
    probe_results[probe] = {
        "primary_method": method,
        "midpoint_mae_m_per_s": mae,
        "observed_completion": completion,
        "gaussian_reference_completion": expected_completion(high - low, mae),
        "mean_interval_width_m_per_s": float(np.mean(high - low)),
    }

mae_grid = np.linspace(0.002, 0.34, 240)
quantile_probabilities = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
payload = {
    "analysis_status": "post_hoc_diagnostic_without_refitting_or_reselection",
    "reference_probe_for_width_distribution": reference_probe,
    "n_test_worlds": int(reference_widths.size),
    "source_sha256": {
        TARGET_PATH.name: sha256(TARGET_PATH),
        PREDICTION_PATH.name: sha256(PREDICTION_PATH),
        SELECTION_PATH.name: sha256(SELECTION_PATH),
    },
    "interval_widths_m_per_s": reference_widths.tolist(),
    "interval_width_summary_m_per_s": {
        "mean": float(np.mean(reference_widths)),
        "quantile_probabilities": quantile_probabilities,
        "quantiles": np.quantile(reference_widths, quantile_probabilities).tolist(),
    },
    "primary_models": probe_results,
    "gaussian_reference_curve": {
        "assumption": "centered Gaussian errors with sigma = MAE * sqrt(pi/2)",
        "mae_m_per_s": mae_grid.tolist(),
        "expected_completion": [expected_completion(reference_widths, value) for value in mae_grid],
    },
}
with OUTPUT_PATH.open("w") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
print(json.dumps({"output": str(OUTPUT_PATH),
                  "width_summary": payload["interval_width_summary_m_per_s"],
                  "primary_models": probe_results}, indent=2))
