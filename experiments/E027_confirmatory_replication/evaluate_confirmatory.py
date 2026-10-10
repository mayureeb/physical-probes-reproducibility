#!/usr/bin/env python3
"""Open targets, physically replay frozen predictions, and test locked claims."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
sys.path.insert(0, str(SOURCE))
import build_data as build  # noqa: E402
import evaluate as evaluator  # noqa: E402

TEST_SEED = 27227
TEST_COUNT = 4096
PROBES = ("fixed_positive_first", "half_then_full_forward",
          "fixed_negative_first", "multisine")
TOLERANCES = (0.06, 0.10, 0.20)
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 2722701


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(first: np.ndarray, second: np.ndarray,
           rng: np.random.Generator) -> dict:
    delta = first.astype(np.int8) - second.astype(np.int8)
    indices = rng.integers(0, len(delta), size=(BOOTSTRAPS, len(delta)))
    samples = delta[indices].mean(axis=1)
    discordant = int(np.sum(delta != 0))
    first_only = int(np.sum(delta == 1))
    p = 1.0 if discordant == 0 else float(
        binomtest(first_only, discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "first_completion": float(first.mean()),
        "second_completion": float(second.mean()),
        "difference": float(delta.mean()),
        "bootstrap_95_interval": np.quantile(samples, (0.025, 0.975)).tolist(),
        "discordant_worlds": discordant,
        "exact_two_sided_p": p,
    }


def holm(records: list[tuple[str, dict]]) -> None:
    ordered = sorted(records, key=lambda item: item[1]["exact_two_sided_p"])
    running = 0.0
    count = len(ordered)
    for rank, (_, record) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * record["exact_two_sided_p"])
        running = max(running, adjusted)
        record["holm_adjusted_p"] = running


def main() -> None:
    prediction_path = HERE / "audit/predictions_before_targets.npz"
    lock = json.loads((HERE / "audit/prediction_lock.json").read_text())
    if sha256(prediction_path) != lock["prediction_sha256"]:
        raise RuntimeError("prediction hash changed after target access")
    predictions = load(prediction_path)
    inputs = load(HERE / "data/test_inputs.npz")
    targets = load(HERE / "audit/test_targets.npz")
    if not np.array_equal(inputs["world_seed"], targets["world_seed"]):
        raise RuntimeError("test target alignment failed")
    action_model = joblib.load(HERE / "checkpoints/half_then_full_action.joblib")
    predictions["half_then_full_forward__true_parameter_action"] = action_model.predict(
        np.concatenate((targets["parameters"], inputs["goal"][:, None]), axis=1)).astype(np.float32)

    complete = build.probes.library()
    build.probes.library = lambda: {name: complete[name] for name in PROBES}
    build.SPLITS["test"] = (TEST_SEED, TEST_COUNT)
    evaluator.PROBES = PROBES
    physical = evaluator.physical_predictions(predictions, {})
    errors = {label: np.abs(values["distance"] - inputs["goal"])
              for label, values in physical.items()}
    arrays = {f"{label}__landing_error": value.astype(np.float32)
              for label, value in errors.items()}
    np.savez_compressed(HERE / "audit/physical_replay.npz", **arrays)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = {}
    locked = []
    for tolerance in (0.10, 0.20):
        for probe, short in (("fixed_positive_first", "positive"),
                             ("fixed_negative_first", "negative")):
            name = f"{short}_minus_multisine_at_{tolerance:.2f}m"
            record = paired(errors[f"{probe}__primary"] <= tolerance,
                            errors["multisine__primary"] <= tolerance, rng)
            comparisons[name] = record
            locked.append((name, record))
    record = paired(errors["multisine__samples_17"] <= 0.06,
                    errors["multisine__samples_9"] <= 0.06, rng)
    comparisons["multisine_17_minus_9_samples"] = record
    locked.append(("multisine_17_minus_9_samples", record))
    record = paired(errors["fixed_positive_first__samples_9"] <= 0.06,
                    errors["fixed_positive_first__samples_401"] <= 0.06, rng)
    comparisons["positive_9_minus_401_samples"] = record
    locked.append(("positive_9_minus_401_samples", record))
    record = paired(errors["half_then_full_forward__true_parameter_action"] <= 0.06,
                    errors["half_then_full_forward__structured"] <= 0.06, rng)
    comparisons["true_parameters_minus_structured"] = record
    locked.append(("true_parameters_minus_structured", record))
    holm(locked)

    descriptive = {label: {
        "mean_landing_error_m": float(error.mean()),
        "completion_by_tolerance": {
            str(value): float(np.mean(error <= value)) for value in TOLERANCES}}
        for label, error in errors.items()}
    result = {
        "status": "confirmatory_replication_complete",
        "worlds": TEST_COUNT,
        "test_seed": TEST_SEED,
        "prediction_sha256": sha256(prediction_path),
        "target_sha256": sha256(HERE / "audit/test_targets.npz"),
        "comparisons": comparisons,
        "all_locked_hypotheses_confirmed": all(
            record["difference"] > 0 and record["holm_adjusted_p"] < 0.05
            for _, record in locked),
        "descriptive": descriptive,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
