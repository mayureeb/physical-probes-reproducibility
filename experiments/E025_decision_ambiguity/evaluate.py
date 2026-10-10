#!/usr/bin/env python3
"""Open E025 targets once, certify ambiguity, and physically evaluate decisions."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr

import build_data as build

HERE = Path(__file__).resolve().parent
BOOTSTRAP_SEED = 2502501
BOOTSTRAP_REPLICATES = 20_000
REPLAY_PER_PROBE = 32
PROBES = ("passive", "fixed_positive_first", "half_then_full_forward",
          "fixed_negative_first", "multisine", "random_piecewise")
TRACE_INDEX = np.arange(0, 401, 50)
VELOCITY_SIGMA = 0.01
FORCE_SIGMA = 0.04
FORCE_RHO = 0.8


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def covariance_inverse() -> tuple[np.ndarray, np.ndarray]:
    length = len(TRACE_INDEX)
    force = FORCE_SIGMA ** 2 * FORCE_RHO ** np.abs(
        np.arange(length)[:, None] - np.arange(length)[None, :])
    return np.eye(length) / VELOCITY_SIGMA ** 2, np.linalg.inv(force)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else 0.0


def physical_completion(distance: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Inherited E019--E023 completion: terminal landing error within tolerance."""
    return np.abs(distance - goal) <= build.e020.e014.TOLERANCE


def ambiguity(probe: str, reference: dict, target: dict) -> tuple[np.ndarray, np.ndarray]:
    ref_trace = reference[f"{probe}__trace"][:, TRACE_INDEX]
    test_trace = target[f"{probe}__trace"][:, TRACE_INDEX]
    ref_low, ref_high = reference[f"{probe}__interval_low"], reference[f"{probe}__interval_high"]
    test_low, test_high = target[f"{probe}__interval_low"], target[f"{probe}__interval_high"]
    velocity_inv, force_inv = covariance_inverse()
    scores = np.zeros(len(test_trace), dtype=np.float64)
    neighbors = np.full(len(test_trace), -1, dtype=np.int32)
    for index in range(len(test_trace)):
        same_goal = reference["goal"] == target["goal"][index]
        conflict = (ref_high < test_low[index]) | (ref_low > test_high[index])
        candidates = np.flatnonzero(same_goal & conflict)
        if not len(candidates):
            continue
        velocity_delta = ref_trace[candidates, :, 0] - test_trace[index, :, 0]
        force_delta = ref_trace[candidates, :, 1] - test_trace[index, :, 1]
        squared = np.einsum("ni,ij,nj->n", velocity_delta, velocity_inv, velocity_delta)
        squared += np.einsum("ni,ij,nj->n", force_delta, force_inv, force_delta)
        chosen = int(np.argmin(squared))
        neighbors[index] = int(candidates[chosen])
        scores[index] = float(ndtr(-np.sqrt(max(0.0, squared[chosen])) / 2.0))
    return scores, neighbors


def physical_predictions(predictions: dict, selection: dict) -> dict:
    worlds = build.make_worlds(build.SPLITS["test"][0], build.SPLITS["test"][1])
    methods = {probe: [key.split("__", 1)[1] for key in predictions
                       if key.startswith(f"{probe}__")] for probe in PROBES}
    output = {f"{probe}__{method}": {"distance": [], "stopped": [], "attained": []}
              for probe in PROBES for method in methods[probe]}
    for start in range(0, len(worlds), 16):
        chunk = worlds[start:start + 16]
        _, models, initial = build.e020.make_models(chunk)
        params = build.e020.parameters(chunk)
        final_by_probe = {}
        for probe, target in build.probes.library().items():
            final_by_probe[probe], _ = build.batch_probe_with_cost(
                models, initial, params, jp.asarray(target))
        blocks, decisions, labels = [], [], []
        for probe in PROBES:
            for method in methods[probe]:
                blocks.append(final_by_probe[probe])
                decisions.append(predictions[f"{probe}__{method}"][start:start + len(chunk)])
                labels.append(f"{probe}__{method}")
        flat_data = build.concatenate_trees([item[0] for item in blocks])
        flat_z1 = jp.concatenate([item[1] for item in blocks])
        flat_z2 = jp.concatenate([item[2] for item in blocks])
        flat_models = build.repeat_tree(models, len(blocks))
        flat_params = build.repeat_params(params, len(blocks))
        flat_decisions = jp.asarray(np.concatenate(decisions), dtype=jp.float32)
        outcome = build.e021.batch_release(flat_models, flat_data, flat_params,
                                           flat_z1, flat_z2, flat_decisions)
        jax.block_until_ready(outcome[0])
        count = len(chunk)
        for block_index, label in enumerate(labels):
            section = slice(block_index * count, (block_index + 1) * count)
            output[label]["distance"].append(np.asarray(outcome[0][section]))
            output[label]["attained"].append(np.asarray(outcome[4][section]))
            output[label]["stopped"].append(np.asarray(outcome[5][section]))
    return {label: {key: np.concatenate(parts) for key, parts in values.items()}
            for label, values in output.items()}


def replay_witnesses(probe: str, neighbors: np.ndarray, scores: np.ndarray,
                     reference: dict, targets: dict) -> dict:
    retained = np.flatnonzero((neighbors >= 0) & (scores > 0))
    retained = retained[np.argsort(scores[retained])[-REPLAY_PER_PROBE:]]
    if not len(retained):
        return {"retained": 0, "validated": 0, "fraction": 1.0}
    calibration_worlds = build.make_worlds(*build.SPLITS["calibration"])
    validation_worlds = build.make_worlds(*build.SPLITS["validation"])
    reference_worlds = calibration_worlds + validation_worlds
    test_worlds = build.make_worlds(*build.SPLITS["test"])
    paired_worlds = [world for index in retained
                     for world in (test_worlds[index], reference_worlds[neighbors[index]])]
    _, models, initial = build.e020.make_models(paired_worlds)
    params = build.e020.parameters(paired_worlds)
    final, _ = build.batch_probe_with_cost(models, initial, params,
                                            jp.asarray(build.probes.library()[probe]))
    data, z1, z2, *_ = final
    goals = jp.asarray([world.goal for world in paired_worlds], dtype=jp.float32)
    low = build.boundary(models, data, params, z1, z2,
                         jp.maximum(0.0, goals - build.e020.e014.TOLERANCE))
    high = build.boundary(models, data, params, z1, z2,
                          goals + build.e020.e014.TOLERANCE)
    low, high = np.asarray(low), np.asarray(high)
    validated = 0
    for pair_index, test_index in enumerate(retained):
        ref_index = neighbors[test_index]
        numerical = (abs(low[2 * pair_index] - targets[f"{probe}__interval_low"][test_index]) <= 0.01
            and abs(high[2 * pair_index] - targets[f"{probe}__interval_high"][test_index]) <= 0.01
            and abs(low[2 * pair_index + 1] - reference[f"{probe}__interval_low"][ref_index]) <= 0.01
            and abs(high[2 * pair_index + 1] - reference[f"{probe}__interval_high"][ref_index]) <= 0.01)
        conflicting = high[2 * pair_index] < low[2 * pair_index + 1] or high[2 * pair_index + 1] < low[2 * pair_index]
        validated += int(numerical and conflicting)
    return {"retained": int(len(retained)), "validated": validated,
            "fraction": float(validated / len(retained)),
            "test_indices": retained.tolist(), "reference_indices": neighbors[retained].tolist()}


def main() -> None:
    started = time.time()
    audit = HERE / "audit"
    selection_path = audit / "selection.json"
    prediction_path = audit / "predictions_pre_evaluation.npz"
    selection, predictions = json.loads(selection_path.read_text()), load(prediction_path)
    calibration, validation = load(HERE / "data/calibration.npz"), load(HERE / "data/validation.npz")
    inputs, targets = load(HERE / "data/test_inputs.npz"), load(audit / "test_targets.npz")
    if not np.array_equal(inputs["world_seed"], targets["world_seed"]):
        raise RuntimeError("test target alignment failed")
    reference = {key: np.concatenate((calibration[key], validation[key]))
                 for key in calibration if key in validation}
    physical = physical_predictions(predictions, selection)
    result = {"status": "e025_evaluated_after_completion_correction", "environment": {
        "devices": [str(device) for device in jax.devices()]}, "probes": {},
        "input_sha256": {"selection": sha256(selection_path),
                         "predictions": sha256(prediction_path),
                         "targets": sha256(audit / "test_targets.npz")}}
    pooled_scores, pooled_failure = [], []
    replay_fractions = []
    for probe in PROBES:
        scores, neighbors = ambiguity(probe, reference, {**inputs, **targets})
        primary = selection["probes"][probe]["primary_method"]
        prediction = predictions[f"{probe}__{primary}"]
        low, high = targets[f"{probe}__interval_low"], targets[f"{probe}__interval_high"]
        decision_failure = ~((prediction >= low) & (prediction <= high))
        pooled_scores.append(scores)
        pooled_failure.append(decision_failure.astype(float))
        replay = replay_witnesses(probe, neighbors, scores, reference, targets)
        replay_fractions.append(replay["fraction"])
        methods = {}
        for label, values in physical.items():
            if not label.startswith(f"{probe}__"):
                continue
            method = label.split("__", 1)[1]
            error = np.abs(values["distance"] - inputs["goal"])
            success = physical_completion(values["distance"], inputs["goal"])
            methods[method] = {"completion": float(np.mean(success)),
                               "completed": int(np.sum(success)),
                               "landing_error_mean_m": float(error.mean()),
                               "target_attained_fraction": float(np.mean(values["attained"])),
                               "stopped_fraction": float(np.mean(values["stopped"]))}
        test_cost_violations = {
            "path": int(np.sum(inputs[f"{probe}__path_displacement"] >
                               selection["caps"]["path_displacement"] + 1e-6)),
            "work": int(np.sum(inputs[f"{probe}__absolute_work"] >
                               selection["caps"]["absolute_work"] + 1e-6))}
        result["probes"][probe] = {"primary_method": primary, "methods": methods,
            "ambiguity_mean": float(scores.mean()),
            "fraction_ambiguity_at_least_0p10": float(np.mean(scores >= 0.10)),
            "witnesses_found": int(np.sum(neighbors >= 0)), "replay": replay,
            "costs": selection["costs"][probe],
            "test_cost_violations": test_cost_violations}
    pooled_scores = np.stack(pooled_scores, axis=1)
    pooled_failure = np.stack(pooled_failure, axis=1)
    point = safe_spearman(pooled_scores.ravel(), pooled_failure.ravel())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        chosen = rng.integers(0, len(inputs["goal"]), len(inputs["goal"]))
        bootstrap[index] = safe_spearman(pooled_scores[chosen].ravel(),
                                         pooled_failure[chosen].ravel())
    ci = np.nanquantile(bootstrap, [0.025, 0.975]).tolist()
    fixed_fraction = result["probes"]["fixed_positive_first"]["fraction_ambiguity_at_least_0p10"]
    candidates = {probe: values["fraction_ambiguity_at_least_0p10"]
                  for probe, values in result["probes"].items()
                  if probe != "passive" and not values["costs"]["path_cap_violations"]
                  and not values["costs"]["work_cap_violations"]
                  and not values["test_cost_violations"]["path"]
                  and not values["test_cost_violations"]["work"]}
    best_probe = min(candidates, key=candidates.get) if candidates else None
    reduction = (0.0 if fixed_fraction == 0 or best_probe is None else
                 1 - candidates[best_probe] / fixed_fraction)
    certificate_pass = min(replay_fractions) >= 0.99 and point >= 0.40 and ci[0] > 0
    action_pass = reduction >= 0.50
    result["certificate_gate"] = {"spearman": point, "ci95": ci,
        "replay_fraction_min": min(replay_fractions), "passed": certificate_pass,
        "bootstrap_seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}
    result["action_comparison_gate"] = {"best_probe": best_probe,
        "ambiguity_fraction_reduction_vs_fixed": reduction, "passed": action_pass}
    result["d_optimal_alias"] = selection["d_optimal_probe"]
    result["gate_passed"] = bool(certificate_pass and action_pass)
    result["elapsed_seconds"] = time.time() - started
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"certificate_gate": result["certificate_gate"],
                      "action_comparison_gate": result["action_comparison_gate"],
                      "gate_passed": result["gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
