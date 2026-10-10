#!/usr/bin/env python3
"""Plan in estimated worlds and freeze actions before test-parameter access."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
sys.path.insert(0, str(SOURCE))
import build_data as build  # noqa: E402

PROBES = ("half_then_full_forward", "fixed_negative_first")
PARAMETERS = ("mass", "mu", "damping", "gain", "strength", "tau1", "tau2", "mix")
PLANNING_BATCH_SIZE = int(os.environ.get("PHYSWORLD_PLANNING_BATCH_SIZE", "256"))


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worlds(parameters: np.ndarray, goals: np.ndarray, seeds: np.ndarray) -> list:
    return [build.e020.e014.World(
        family=4, mass=float(row[0]), mu=float(row[1]), damping=float(row[2]),
        goal=float(goal), gain=float(row[3]), strength=float(row[4]),
        tau1=float(row[5]), tau2=float(row[6]), mix=float(row[7]), seed=int(seed))
        for row, goal, seed in zip(parameters, goals, seeds)]


def planned_actions(probe: str, parameters: np.ndarray,
                    goals: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    target = build.probes.library()[probe]
    actions = []
    started = time.perf_counter()
    for start in range(0, len(goals), PLANNING_BATCH_SIZE):
        chunk = worlds(parameters[start:start + PLANNING_BATCH_SIZE],
                       goals[start:start + PLANNING_BATCH_SIZE],
                       seeds[start:start + PLANNING_BATCH_SIZE])
        _, models, initial = build.e020.make_models(chunk)
        params = build.e020.parameters(chunk)
        final, _ = build.batch_probe_with_cost(models, initial, params, build.jp.asarray(target))
        data, z1, z2, *_ = final
        speed = build.boundary(models, data, params, z1, z2,
                               build.jp.asarray(goals[start:start + len(chunk)]))
        jax.block_until_ready(speed)
        actions.append(np.asarray(speed, dtype=np.float32))
        print(json.dumps({"probe": probe, "planned": start + len(chunk),
                          "count": len(goals),
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    return np.concatenate(actions)


def main() -> None:
    source_path = HERE / "audit/estimates_and_actions_before_targets.npz"
    lock = json.loads((HERE / "audit/estimation_lock.json").read_text())
    if sha256(source_path) != lock["artifact_sha256"]:
        raise RuntimeError("estimation artifact hash mismatch")
    source = load(source_path)
    inputs = load(HERE / "data/test_inputs.npz")
    output = {key: value for key, value in source.items() if not key.endswith("estimated_parameters")}
    for probe in PROBES:
        output[f"{probe}__estimated_planner"] = planned_actions(
            probe, source[f"{probe}__estimated_parameters"], inputs["goal"], inputs["world_seed"])
    output_path = HERE / "audit/actions_before_targets.npz"
    np.savez_compressed(output_path, **output)
    record = {
        "status": "trace_conditioned_actions_frozen_before_target_access",
        "estimation_artifact_sha256": sha256(source_path),
        "action_artifact_sha256": sha256(output_path),
        "test_targets_accessed": False,
    }
    (HERE / "audit/action_lock.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
