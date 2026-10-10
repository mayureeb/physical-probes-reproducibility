#!/usr/bin/env python3
"""Plan from estimated state and freeze every trace-conditioned test action."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
sys.path.insert(0, str(SOURCE))
import build_data as build  # noqa: E402

PROBES = ("decision_directed", "d_optimal", "fixed_positive_first",
          "fixed_negative_first")


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worlds(parameters: np.ndarray, goals: np.ndarray, seeds: np.ndarray) -> list:
    return [build.e020.e014.World(family=4, mass=float(p[0]), mu=float(p[1]),
        damping=float(p[2]), goal=float(g), gain=float(p[3]), strength=float(p[4]),
        tau1=float(p[5]), tau2=float(p[6]), mix=float(p[7]), seed=int(s))
        for p, g, s in zip(parameters, goals, seeds)]


def plan(waveform: np.ndarray, parameters: np.ndarray,
         goals: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    actions = []
    started = time.perf_counter()
    for start in range(0, len(goals), 256):
        chunk = worlds(parameters[start:start + 256], goals[start:start + 256],
                       seeds[start:start + 256])
        _, models, initial = build.e020.make_models(chunk)
        params = build.e020.parameters(chunk)
        final, _ = build.batch_probe_with_cost(models, initial, params,
                                                build.jp.asarray(waveform))
        data, z1, z2, *_ = final
        speed = build.boundary(models, data, params, z1, z2,
                               build.jp.asarray(goals[start:start + len(chunk)]))
        jax.block_until_ready(speed)
        actions.append(np.asarray(speed, dtype=np.float32))
        print(json.dumps({"planned": start + len(chunk), "worlds": len(goals),
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    return np.concatenate(actions)


def main() -> None:
    source_path = HERE / "audit/estimates_and_direct_actions.npz"
    lock = json.loads((HERE / "audit/estimation_lock.json").read_text())
    if sha256(source_path) != lock["artifact_sha256"] or lock["test_targets_accessed"]:
        raise RuntimeError("estimation boundary failed")
    source = load(source_path)
    inputs = load(HERE / "data/test_inputs.npz")
    waveforms = load(HERE / "audit/selected_waveforms.npz")
    output = {key: value for key, value in source.items()
              if not key.endswith("estimated_parameters")}
    for probe in PROBES:
        output[f"{probe}__estimated_planner"] = plan(
            waveforms[probe], source[f"{probe}__estimated_parameters"],
            inputs["goal"], inputs["world_seed"])
    path = HERE / "audit/actions_before_targets.npz"
    np.savez_compressed(path, **output)
    record = {"status": "all_trace_conditioned_actions_frozen_before_test_targets",
              "estimation_artifact_sha256": sha256(source_path),
              "action_artifact_sha256": sha256(path),
              "test_targets_accessed": False}
    (HERE / "audit/action_lock.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

