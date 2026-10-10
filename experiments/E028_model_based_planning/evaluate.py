#!/usr/bin/env python3
"""Open test parameters, compute the true planner, and replay locked actions."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
sys.path.insert(0, str(SOURCE))
import build_data as build  # noqa: E402
import evaluate as physical_evaluator  # noqa: E402
from plan_before_targets import planned_actions  # noqa: E402

TEST_SEED = 28228
TEST_COUNT = 2048
PROBES = ("half_then_full_forward", "fixed_negative_first")
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 2822801


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(first: np.ndarray, second: np.ndarray,
           rng: np.random.Generator) -> dict:
    delta = first.astype(np.int8) - second.astype(np.int8)
    samples = np.empty(BOOTSTRAPS, dtype=np.float32)
    for start in range(0, BOOTSTRAPS, 500):
        indices = rng.integers(0, len(delta), size=(min(500, BOOTSTRAPS - start), len(delta)))
        samples[start:start + len(indices)] = delta[indices].mean(axis=1)
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


def holm(records: list[dict]) -> None:
    ordered = sorted(records, key=lambda record: record["exact_two_sided_p"])
    running = 0.0
    count = len(ordered)
    for rank, record in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * record["exact_two_sided_p"]))
        record["holm_adjusted_p"] = running


def main() -> None:
    action_path = HERE / "audit/actions_before_targets.npz"
    action_lock = json.loads((HERE / "audit/action_lock.json").read_text())
    if sha256(action_path) != action_lock["action_artifact_sha256"]:
        raise RuntimeError("frozen action hash mismatch")
    actions = load(action_path)
    inputs = load(HERE / "data/test_inputs.npz")
    targets = load(HERE / "audit/test_targets.npz")
    if not np.array_equal(inputs["world_seed"], targets["world_seed"]):
        raise RuntimeError("test target alignment failed")

    for probe in PROBES:
        actions[f"{probe}__true_planner"] = planned_actions(
            probe, targets["parameters"], inputs["goal"], inputs["world_seed"])

    complete = build.probes.library()
    build.probes.library = lambda: {name: complete[name] for name in PROBES}
    build.SPLITS["test"] = (TEST_SEED, TEST_COUNT)
    physical_evaluator.PROBES = PROBES
    physical = physical_evaluator.physical_predictions(actions, {})
    errors = {label: np.abs(values["distance"] - inputs["goal"])
              for label, values in physical.items()}
    np.savez_compressed(HERE / "audit/physical_replay.npz", **{
        f"{label}__landing_error": error.astype(np.float32)
        for label, error in errors.items()})

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = {}
    records = []
    for probe in PROBES:
        record = paired(errors[f"{probe}__estimated_planner"] <= 0.06,
                        errors[f"{probe}__learned_action"] <= 0.06, rng)
        comparisons[f"{probe}__estimated_planner_minus_learned_action"] = record
        records.append(record)
        record = paired(errors[f"{probe}__true_planner"] <= 0.06,
                        errors[f"{probe}__estimated_planner"] <= 0.06, rng)
        comparisons[f"{probe}__true_minus_estimated_planner"] = record
        records.append(record)
    holm(records)
    result = {
        "status": "model_based_planning_evaluated",
        "worlds": TEST_COUNT,
        "test_seed": TEST_SEED,
        "action_artifact_sha256": sha256(action_path),
        "target_sha256": sha256(HERE / "audit/test_targets.npz"),
        "comparisons": comparisons,
        "all_locked_hypotheses_confirmed": all(
            record["difference"] > 0
            and record["bootstrap_95_interval"][0] > 0
            and record["holm_adjusted_p"] < 0.05 for record in records),
        "descriptive": {label: {
            "completion": float(np.mean(error <= 0.06)),
            "mean_landing_error_m": float(error.mean())}
            for label, error in errors.items()},
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
