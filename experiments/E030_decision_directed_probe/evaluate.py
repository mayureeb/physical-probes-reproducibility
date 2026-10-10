#!/usr/bin/env python3
"""Open test targets after action freeze and evaluate the registered contrasts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jax
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
sys.path.insert(0, str(SOURCE))
import build_data as build  # noqa: E402

PROBES = ("decision_directed", "d_optimal", "fixed_positive_first",
          "fixed_negative_first")
BOOTSTRAP_SEED = 303030
BOOTSTRAP_REPLICATES = 20_000


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(actions: dict[str, np.ndarray], waveforms: dict[str, np.ndarray]) -> dict:
    worlds = build.make_worlds(30330, 2048)
    output = {f"{probe}__{method}": {"distance": [], "success": []}
              for probe in PROBES for method in ("direct", "estimated_planner", "true_planner")}
    costs = {probe: {"path": [], "work": [], "peak": []} for probe in PROBES}
    for start in range(0, len(worlds), 64):
        chunk = worlds[start:start + 64]
        _, models, initial = build.e020.make_models(chunk)
        params = build.e020.parameters(chunk)
        goals = np.asarray([world.goal for world in chunk], dtype=np.float32)
        for probe in PROBES:
            final, _ = build.batch_probe_with_cost(models, initial, params,
                build.jp.asarray(waveforms[probe]))
            data, z1, z2, path, work, peak = final
            true_action = build.boundary(models, data, params, z1, z2,
                                         build.jp.asarray(goals))
            choices = [actions[f"{probe}__direct"][start:start + len(chunk)],
                       actions[f"{probe}__estimated_planner"][start:start + len(chunk)],
                       np.asarray(true_action)]
            flat_data = build.repeat_tree(data, 3)
            flat_z1 = build.jp.concatenate([z1] * 3)
            flat_z2 = build.jp.concatenate([z2] * 3)
            flat_models = build.repeat_tree(models, 3)
            flat_params = build.repeat_params(params, 3)
            outcome = build.e021.batch_release(flat_models, flat_data, flat_params,
                flat_z1, flat_z2, build.jp.asarray(np.concatenate(choices), dtype=build.jp.float32))
            jax.block_until_ready(outcome[0])
            for index, method in enumerate(("direct", "estimated_planner", "true_planner")):
                section = slice(index * len(chunk), (index + 1) * len(chunk))
                distance = np.asarray(outcome[0][section])
                label = f"{probe}__{method}"
                output[label]["distance"].append(distance)
                output[label]["success"].append(np.abs(distance - goals) <= build.e020.e014.TOLERANCE)
            costs[probe]["path"].append(np.asarray(path))
            costs[probe]["work"].append(np.asarray(work))
            costs[probe]["peak"].append(np.asarray(peak))
        print(json.dumps({"evaluated": start + len(chunk), "worlds": len(worlds)}), flush=True)
    packed = {label: {key: np.concatenate(parts) for key, parts in values.items()}
              for label, values in output.items()}
    packed["costs"] = {probe: {key: np.concatenate(parts) for key, parts in values.items()}
                       for probe, values in costs.items()}
    return packed


def contrast(first: np.ndarray, second: np.ndarray, seed: int) -> dict:
    first, second = first.astype(bool), second.astype(bool)
    delta = first.astype(float) - second.astype(float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(delta), (BOOTSTRAP_REPLICATES, len(delta)))
    bootstrap = delta[index].mean(1)
    wins = int(np.sum(first & ~second))
    losses = int(np.sum(~first & second))
    p = float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue) \
        if wins + losses else 1.0
    return {"first_completion": float(first.mean()), "second_completion": float(second.mean()),
            "difference": float(delta.mean()),
            "bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "discordant_worlds": wins + losses, "directional_exact_p": p}


def holm(records: dict[str, dict]) -> None:
    ordered = sorted(records, key=lambda key: records[key]["directional_exact_p"])
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        adjusted = min(1.0, records[key]["directional_exact_p"] * (total - rank))
        running = max(running, adjusted)
        records[key]["holm_adjusted_p"] = running


def main() -> None:
    action_path = HERE / "audit/actions_before_targets.npz"
    action_lock = json.loads((HERE / "audit/action_lock.json").read_text())
    if sha256(action_path) != action_lock["action_artifact_sha256"]:
        raise RuntimeError("action artifact hash mismatch")
    actions = load(action_path)
    targets = load(HERE / "audit/test_targets.npz")
    inputs = load(HERE / "data/test_inputs.npz")
    if not np.array_equal(inputs["world_seed"], targets["world_seed"]):
        raise RuntimeError("test target alignment failed")
    waveforms = load(HERE / "audit/selected_waveforms.npz")
    physical = replay(actions, waveforms)
    descriptive = {}
    for probe in PROBES:
        for method in ("direct", "estimated_planner", "true_planner"):
            item = physical[f"{probe}__{method}"]
            descriptive[f"{probe}__{method}"] = {
                "completion": float(item["success"].mean()),
                "mean_landing_error_m": float(np.mean(np.abs(item["distance"] - inputs["goal"]))) }
    best_fixed = max(("fixed_positive_first", "fixed_negative_first"),
                     key=lambda probe: descriptive[f"{probe}__direct"]["completion"])
    comparisons = {
        "decision_minus_d_optimal_direct": contrast(
            physical["decision_directed__direct"]["success"],
            physical["d_optimal__direct"]["success"], BOOTSTRAP_SEED),
        "decision_minus_best_fixed_direct": contrast(
            physical["decision_directed__direct"]["success"],
            physical[f"{best_fixed}__direct"]["success"], BOOTSTRAP_SEED + 1),
        "decision_minus_d_optimal_planner": contrast(
            physical["decision_directed__estimated_planner"]["success"],
            physical["d_optimal__estimated_planner"]["success"], BOOTSTRAP_SEED + 2),
        "decision_true_minus_estimated_planner": contrast(
            physical["decision_directed__true_planner"]["success"],
            physical["decision_directed__estimated_planner"]["success"], BOOTSTRAP_SEED + 3),
    }
    holm(comparisons)
    selection = json.loads((HERE / "audit/selection_lock.json").read_text())
    caps = selection["caps"]
    costs = {}
    for probe in PROBES:
        values = physical["costs"][probe]
        costs[probe] = {"path_violations": int(np.sum(values["path"] > caps["path_displacement"] + 1e-6)),
                        "work_violations": int(np.sum(values["work"] > caps["absolute_work"] + 1e-6)),
                        "peak_speed_max": float(values["peak"].max())}
    gates = {
        "decision_vs_d_optimal": bool(comparisons["decision_minus_d_optimal_direct"]["difference"] >= 0.05
            and comparisons["decision_minus_d_optimal_direct"]["bootstrap_95_interval"][0] > 0),
        "decision_vs_best_fixed": bool(comparisons["decision_minus_best_fixed_direct"]["difference"] >= 0.03
            and comparisons["decision_minus_best_fixed_direct"]["bootstrap_95_interval"][0] > 0),
        "structured_vs_d_optimal": bool(comparisons["decision_minus_d_optimal_planner"]["bootstrap_95_interval"][0] > 0),
        "true_state_positive_control": bool(comparisons["decision_true_minus_estimated_planner"]["bootstrap_95_interval"][0] > 0),
    }
    result = {"status": "decision_directed_probe_evaluated", "worlds": 2048,
              "test_seed": 30330, "selection": {
                  "decision_directed_source": selection["decision_directed_source"],
                  "d_optimal_source": selection["d_optimal_source"],
                  "best_fixed_on_test": best_fixed},
              "comparisons": comparisons, "registered_gates": gates,
              "descriptive": descriptive, "costs": costs,
              "action_artifact_sha256": sha256(action_path),
              "target_sha256": sha256(HERE / "audit/test_targets.npz")}
    output = HERE / "results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

