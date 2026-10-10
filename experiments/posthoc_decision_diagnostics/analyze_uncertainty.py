#!/usr/bin/env python3
"""Paired bootstrap intervals for the post hoc decision diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
PROBES = ("passive", "fixed_positive_first", "half_then_full_forward",
          "fixed_negative_first", "multisine", "random_piecewise")
SAMPLE_COUNTS = (9, 17, 41, 81, 401)
TOLERANCES_M = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20)
BOOTSTRAP_DRAWS = 20_000
SEED = 29029


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def interval_success(prediction: np.ndarray, targets: dict, probe: str) -> np.ndarray:
    return ((prediction >= targets[f"{probe}__interval_low"])
            & (prediction <= targets[f"{probe}__interval_high"]))


def paired_interval(first: np.ndarray, second: np.ndarray,
                    rng: np.random.Generator) -> dict:
    """Return first minus second in percentage points with a paired percentile CI."""
    difference = first.astype(float) - second.astype(float)
    counts = rng.multinomial(len(difference), np.full(len(difference), 1 / len(difference)),
                             size=BOOTSTRAP_DRAWS)
    samples = counts @ difference / len(difference) * 100
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "difference_percentage_points": float(difference.mean() * 100),
        "bootstrap_95_interval_percentage_points": [float(low), float(high)],
        "paired_worlds": int(len(difference)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    posthoc = load(HERE / "predictions.npz")
    structured = load(HERE / "structured_decoder_predictions.npz")
    structured_physical = load(HERE / "structured_physical_predictions.npz")
    primary = load(SOURCE / "audit/predictions_pre_evaluation.npz")
    targets = load(SOURCE / "audit/test_targets.npz")
    selection = json.loads((SOURCE / "audit/selection.json").read_text())

    primary_success = {}
    for probe in PROBES:
        method = selection["probes"][probe]["primary_method"]
        primary_success[probe] = interval_success(primary[f"{probe}__{method}"], targets, probe)

    tolerance = {}
    for value in TOLERANCES_M:
        key = str(value)
        multisine = posthoc["multisine__primary__landing_error"] <= value
        tolerance[key] = {
            "positive_first_minus_multisine": paired_interval(
                posthoc["fixed_positive_first__primary__landing_error"] <= value,
                multisine, rng),
            "negative_first_minus_multisine": paired_interval(
                posthoc["fixed_negative_first__primary__landing_error"] <= value,
                multisine, rng),
        }

    sampling = {}
    for probe in ("fixed_positive_first", "multisine"):
        base = interval_success(posthoc[f"{probe}__samples_9"], targets, probe)
        sampling[probe] = {}
        for count in SAMPLE_COUNTS[1:]:
            current = interval_success(posthoc[f"{probe}__samples_{count}"], targets, probe)
            sampling[probe][f"{count}_minus_9_samples"] = paired_interval(current, base, rng)

    decoders = {}
    for probe in PROBES:
        pool = interval_success(posthoc[f"{probe}__pool_bayes_calibrated"], targets, probe)
        structured_success = structured_physical[
            f"{probe}__structured__landing_error"] <= 0.06
        true_parameter = structured_physical[
            f"{probe}__true_parameter_action__landing_error"] <= 0.06
        decoders[probe] = {
            "pool_bayes_minus_primary": paired_interval(pool, primary_success[probe], rng),
            "structured_minus_primary": paired_interval(
                structured_success, primary_success[probe], rng),
            "true_parameter_action_minus_structured": paired_interval(
                true_parameter, structured_success, rng),
        }

    payload = {
        "status": "post_hoc_uncertainty_complete",
        "method": "paired percentile bootstrap over the 256 held-out worlds",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "tolerance_comparisons": tolerance,
        "sampling_rate_comparisons": sampling,
        "decoder_comparisons": decoders,
    }
    (HERE / "uncertainty.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
