#!/usr/bin/env python3
"""Generate fresh paired clean-trace data for E023."""
from __future__ import annotations

import argparse
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
SPLITS = {"train": (23023, 4096), "validation": (23123, 1024), "test": (23223, 1024)}
PARAMETERS = ("mass", "mu", "damping", "gain", "strength", "tau1", "tau2", "mix")
NOISE_SIGMA = 0.04
NOISE_REPLICATES = 16
SENSITIVITY_DELTA = 0.025


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e020 = load_module("e023_e020", HERE.parent / "E020_mjx_full_sequence" / "run.py")
e021 = load_module("e023_e021", HERE.parent / "E021_mjx_full_sequence_acceptance" / "run.py")


def generate(seed: int, count: int, chunk_size: int):
    worlds_all = e020.e014.make_worlds(seed, count, [4])
    traces, goals, speeds, sensitivities, parameters = [], [], [], [], []
    started = time.perf_counter()
    for start in range(0, count, chunk_size):
        worlds = worlds_all[start:start + chunk_size]
        _, models, data = e020.make_models(worlds)
        physical = e020.parameters(worlds)
        probe_final, trace = e020.batch_probe(models, data, physical, jp.asarray(e020.probe_targets()))
        probe_data, z1, z2 = probe_final
        goal = jp.asarray([world.goal for world in worlds], dtype=jp.float32)
        oracle_speed, oracle_outcome = e021.batch_oracle(models, probe_data, physical, z1, z2, goal)
        jax.block_until_ready(oracle_outcome[0])
        low = jp.maximum(0.0, oracle_speed - SENSITIVITY_DELTA)
        high = jp.minimum(3.0, oracle_speed + SENSITIVITY_DELTA)
        low_outcome = e021.batch_release(models, probe_data, physical, z1, z2, low)
        high_outcome = e021.batch_release(models, probe_data, physical, z1, z2, high)
        jax.block_until_ready(high_outcome[0])
        denominator = jp.maximum(high - low, 1e-6)
        local_sensitivity = jp.abs(high_outcome[0] - low_outcome[0]) / denominator
        traces.append(np.asarray(trace, dtype=np.float32))
        goals.append(np.asarray(goal, dtype=np.float32))
        speeds.append(np.asarray(oracle_speed, dtype=np.float32))
        sensitivities.append(np.asarray(local_sensitivity, dtype=np.float32))
        parameters.append(np.asarray([[getattr(world, key) for key in PARAMETERS] for world in worlds], dtype=np.float32))
        print(json.dumps({"built": start + len(worlds), "count": count,
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    return {
        "trace_clean": np.concatenate(traces),
        "goal": np.concatenate(goals),
        "oracle_speed": np.concatenate(speeds),
        "local_sensitivity": np.concatenate(sensitivities),
        "parameters": np.concatenate(parameters),
        "noise_seed": np.asarray(seed + 900, dtype=np.int64),
        "noise_sigma": np.asarray(NOISE_SIGMA, dtype=np.float32),
        "noise_replicates": np.asarray(NOISE_REPLICATES, dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HERE / "data"))
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "e023_generated_data",
        "environment": {"mujoco": mujoco.__version__, "jax": jax.__version__,
                        "devices": [str(device) for device in jax.devices()]},
        "parameter_order": PARAMETERS,
        "noise": {"sigma": NOISE_SIGMA, "replicates": NOISE_REPLICATES,
                  "views": ["clean", "single", "mean4", "mean16"]},
        "splits": {},
    }
    for name, (seed, count) in SPLITS.items():
        started = time.perf_counter()
        arrays = generate(seed, count, args.chunk_size)
        path = root / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        manifest["splits"][name] = {
            "seed": seed, "rows": count,
            "trace_shape": list(arrays["trace_clean"].shape),
            "elapsed_seconds": time.perf_counter() - started,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(json.dumps({"saved": name, **manifest["splits"][name]}), flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
