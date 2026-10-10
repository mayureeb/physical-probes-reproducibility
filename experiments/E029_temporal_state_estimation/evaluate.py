#!/usr/bin/env python3
"""Open test parameters, replay actions, and run the registered tests."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
E025 = HERE.parent / "E025_decision_ambiguity"
E028 = HERE.parent / "E028_model_based_planning"
sys.path.insert(0, str(E025))
sys.path.insert(0, str(E028))
import build_data as build  # noqa: E402
import plan_before_targets as planning  # noqa: E402

spec = importlib.util.spec_from_file_location("e029_physical_evaluator", E025 / "evaluate.py")
physical_evaluator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(physical_evaluator)

PROBE = "fixed_negative_first"
BUDGETS = (9, 33, 101, 401)
TEST_SEED = 29229
TEST_COUNT = 2048
BOOTSTRAPS = 20_000


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
        index = rng.integers(0, len(delta), size=(min(500, BOOTSTRAPS-start), len(delta)))
        samples[start:start + len(index)] = delta[index].mean(1)
    discordant = int(np.sum(delta != 0))
    first_only = int(np.sum(delta == 1))
    p = 1.0 if not discordant else float(binomtest(first_only, discordant).pvalue)
    return {"first_completion": float(first.mean()),
            "second_completion": float(second.mean()),
            "difference": float(delta.mean()),
            "bootstrap_95_interval": np.quantile(samples, (0.025, 0.975)).tolist(),
            "discordant_worlds": discordant, "exact_two_sided_p": p}


def holm(records: list[dict]) -> None:
    ordered = sorted(records, key=lambda item: item["exact_two_sided_p"])
    running = 0.0
    for rank, record in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered)-rank) * record["exact_two_sided_p"]))
        record["holm_adjusted_p"] = running


def main() -> None:
    action_path = HERE / "audit/actions_before_targets.npz"
    action_lock = json.loads((HERE / "audit/action_lock.json").read_text())
    if sha256(action_path) != action_lock["action_artifact_sha256"]:
        raise RuntimeError("action artifact hash mismatch")
    actions = load(action_path)
    inputs = load(HERE / "data/test_inputs.npz")
    targets = load(HERE / "audit/test_targets.npz")
    estimates = load(HERE / "audit/estimates_before_targets.npz")
    if not np.array_equal(inputs["world_seed"], targets["world_seed"]):
        raise RuntimeError("test alignment failed")
    actions["true_state"] = planning.planned_actions(
        PROBE, targets["parameters"], inputs["goal"], inputs["world_seed"])

    library = build.probes.library()
    build.probes.library = lambda: {PROBE: library[PROBE]}
    build.SPLITS["test"] = (TEST_SEED, TEST_COUNT)
    physical_evaluator.PROBES = (PROBE,)
    labeled = {f"{PROBE}__{key}": value for key, value in actions.items()}
    replay = physical_evaluator.physical_predictions(labeled, {})
    errors = {key.split("__", 1)[1]: np.abs(value["distance"] - inputs["goal"])
              for key, value in replay.items()}
    np.savez_compressed(HERE / "audit/physical_replay.npz", **{
        f"{key}__landing_error": value.astype(np.float32) for key, value in errors.items()})

    parameter_scale = targets["parameters"].std(0) + 1e-8
    parameter_nmse = {str(budget): float(np.mean(((
        estimates[f"budget_{budget}__estimated_parameters"] - targets["parameters"]
        ) / parameter_scale) ** 2)) for budget in BUDGETS}
    complete = {key: value <= 0.06 for key, value in errors.items()}
    rng = np.random.default_rng(2922901)
    comparisons = {
        "budget_33_minus_9": paired(complete["budget_33"], complete["budget_9"], rng),
        "budget_401_minus_9": paired(complete["budget_401"], complete["budget_9"], rng),
        "true_state_minus_401": paired(complete["true_state"], complete["budget_401"], rng),
    }
    holm(list(comparisons.values()))
    result = {"status": "temporal_state_estimation_evaluated", "worlds": TEST_COUNT,
        "test_seed": TEST_SEED, "action_artifact_sha256": sha256(action_path),
        "target_sha256": sha256(HERE / "audit/test_targets.npz"),
        "comparisons": comparisons,
        "all_locked_hypotheses_confirmed": all(
            r["difference"] > 0 and r["bootstrap_95_interval"][0] > 0
            and r["holm_adjusted_p"] < 0.05 for r in comparisons.values()),
        "descriptive": {key: {"completion": float(complete[key].mean()),
            "mean_landing_error_m": float(errors[key].mean())} for key in errors},
        "parameter_nmse": parameter_nmse,
        "completion_monotone_across_budgets": bool(np.all(np.diff([
            complete[f"budget_{budget}"].mean() for budget in BUDGETS]) >= 0))}
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
