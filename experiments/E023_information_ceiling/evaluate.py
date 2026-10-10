#!/usr/bin/env python3
"""Physically evaluate E023 frozen predictions on untouched worlds."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e020 = load_module("e023_eval_e020", HERE.parent / "E020_mjx_full_sequence" / "run.py")
e021 = load_module("e023_eval_e021", HERE.parent / "E021_mjx_full_sequence_acceptance" / "run.py")


def evaluate_all(chosen, worlds_all, chunk_size=128):
    outcomes = {name: [] for name in chosen}
    for start in range(0, len(worlds_all), chunk_size):
        worlds = worlds_all[start:start + chunk_size]
        _, models, data = e020.make_models(worlds)
        physical = e020.parameters(worlds)
        probe_final, _ = e020.batch_probe(models, data, physical, jp.asarray(e020.probe_targets()))
        probe_data, z1, z2 = probe_final
        for name, speeds in chosen.items():
            result = e021.batch_release(models, probe_data, physical, z1, z2,
                                        jp.asarray(speeds[start:start + len(worlds)], dtype=jp.float32))
            jax.block_until_ready(result[0])
            outcomes[name].append(np.asarray(result[0], dtype=np.float32))
    return {name: np.concatenate(parts) for name, parts in outcomes.items()}


def metrics(speed, distance, target_speed, goal, sensitivity):
    error = np.abs(distance - goal)
    speed_quartile = np.digitize(target_speed, np.quantile(target_speed, (0.25, 0.5, 0.75)))
    sensitivity_quartile = np.digitize(sensitivity, np.quantile(sensitivity, (0.25, 0.5, 0.75)))
    return {
        "completion": float(np.mean(error <= e020.e014.TOLERANCE)),
        "completed": int(np.sum(error <= e020.e014.TOLERANCE)),
        "landing_error_mean_m": float(np.mean(error)),
        "release_speed_mae_mps": float(np.mean(np.abs(speed - target_speed))),
        "release_speed_bias_mps": float(np.mean(speed - target_speed)),
        "completion_by_target_speed_quartile": [float(np.mean(error[speed_quartile == i] <= e020.e014.TOLERANCE)) for i in range(4)],
        "completion_by_sensitivity_quartile": [float(np.mean(error[sensitivity_quartile == i] <= e020.e014.TOLERANCE)) for i in range(4)],
    }


def main():
    test = np.load(HERE / "data" / "test.npz")
    frozen = np.load(HERE / "audit" / "predictions_pre_evaluation.npz")
    selection = json.loads((HERE / "audit" / "selection.json").read_text())
    chosen = {name: frozen[name].astype(np.float32) for name in frozen.files}
    worlds = e020.e014.make_worlds(23223, len(test["goal"]), [4])
    started = time.time()
    outcomes = evaluate_all(chosen, worlds)
    result_metrics = {name: metrics(speed, outcomes[name], test["oracle_speed"],
                                    test["goal"], test["local_sensitivity"])
                      for name, speed in chosen.items()}
    primary = selection["selection"]["primary_clean_method"]
    gate = {
        "primary_clean_method": primary,
        "primary_clean_completion": result_metrics[primary]["completion"],
        "completion_required": 0.80,
        "oracle_completion": result_metrics["oracle"]["completion"],
        "oracle_required": 0.99,
    }
    gate["passed"] = bool(gate["primary_clean_completion"] >= 0.80 and gate["oracle_completion"] >= 0.99)
    paired = {}
    family = "tree" if primary.startswith("tree") else "gru"
    suffix = "" if family == "tree" else "_ensemble"
    clean_completion = result_metrics[f"{family}_clean{suffix}"]["completion"]
    for view in ("single", "mean4", "mean16"):
        value = result_metrics[f"{family}_{view}{suffix}"]["completion"]
        paired[view] = {"completion": value, "absolute_change_from_clean": value - clean_completion}
    result = {
        "status": "e023_locked_result",
        "environment": {"mujoco": mujoco.__version__, "jax": jax.__version__,
                        "devices": [str(device) for device in jax.devices()]},
        "gate": gate,
        "selected_family_paired_noise_effect": paired,
        "metrics": result_metrics,
        "timing": {"physical_evaluation_seconds": time.time() - started},
        "input_sha256": {name: hashlib.sha256((HERE / "audit" / name).read_bytes()).hexdigest()
                         for name in ("selection.json", "predictions_pre_evaluation.npz")},
    }
    np.savez_compressed(HERE / "audit" / "test_outcomes.npz",
                        **{f"speed_{name}": value for name, value in chosen.items()},
                        **{f"distance_{name}": value for name, value in outcomes.items()})
    (HERE / "audit" / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
