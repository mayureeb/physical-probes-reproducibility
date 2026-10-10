#!/usr/bin/env python3
"""Generate fresh trace inputs and separately stored physical parameters."""
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

TEST_SEED = 28228
TEST_COUNT = 2048
PROBES = ("half_then_full_forward", "fixed_negative_first")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    lock = HERE / "audit/prerun_lock.json"
    if not lock.exists():
        raise RuntimeError("prerun lock is required")
    complete = build.probes.library()
    build.probes.library = lambda: {name: complete[name] for name in PROBES}
    worlds_all = build.make_worlds(TEST_SEED, TEST_COUNT)
    collected: dict[str, list[np.ndarray]] = {}
    started = time.perf_counter()
    for start in range(0, TEST_COUNT, 128):
        worlds = worlds_all[start:start + 128]
        _, models, initial = build.e020.make_models(worlds)
        params = build.e020.parameters(worlds)
        for name, target in build.probes.library().items():
            final, trace = build.batch_probe_with_cost(
                models, initial, params, build.jp.asarray(target))
            jax.block_until_ready(trace)
            _, _, _, path, work, peak = final
            for key, value in {
                f"{name}__trace": np.asarray(trace, dtype=np.float32),
                f"{name}__path_displacement": np.asarray(path, dtype=np.float32),
                f"{name}__absolute_work": np.asarray(work, dtype=np.float32),
                f"{name}__peak_speed": np.asarray(peak, dtype=np.float32),
            }.items():
                collected.setdefault(key, []).append(value)
        print(json.dumps({"built": start + len(worlds), "count": TEST_COUNT,
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    inputs = {key: np.concatenate(parts) for key, parts in collected.items()}
    inputs["goal"] = np.asarray([world.goal for world in worlds_all], dtype=np.float32)
    inputs["world_seed"] = np.asarray([world.seed for world in worlds_all], dtype=np.int64)
    targets = {
        "parameters": np.asarray(
            [[getattr(world, key) for key in build.PARAMETERS] for world in worlds_all],
            dtype=np.float32),
        "goal": inputs["goal"],
        "world_seed": inputs["world_seed"],
    }
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "audit").mkdir(exist_ok=True)
    input_path = HERE / "data/test_inputs.npz"
    target_path = HERE / "audit/test_targets.npz"
    np.savez_compressed(input_path, **inputs)
    np.savez_compressed(target_path, **targets)
    manifest = {
        "status": "test_traces_generated_before_estimation",
        "seed": TEST_SEED,
        "worlds": TEST_COUNT,
        "probes": list(PROBES),
        "elapsed_seconds": time.perf_counter() - started,
        "input_sha256": sha256(input_path),
        "target_sha256": sha256(target_path),
        "prerun_lock_sha256": sha256(lock),
    }
    (HERE / "data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
