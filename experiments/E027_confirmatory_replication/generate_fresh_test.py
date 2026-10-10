#!/usr/bin/env python3
"""Generate the locked fresh test split for the confirmatory replication."""
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

TEST_SEED = 27227
TEST_COUNT = 4096
PROBES = ("fixed_positive_first", "half_then_full_forward",
          "fixed_negative_first", "multisine")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    lock = HERE / "audit/prerun_lock.json"
    if not lock.exists():
        raise RuntimeError("prerun lock must exist before generation")
    complete = build.probes.library()
    build.probes.library = lambda: {name: complete[name] for name in PROBES}
    started = time.perf_counter()
    worlds_all = build.make_worlds(TEST_SEED, TEST_COUNT)
    collected: dict[str, list[np.ndarray]] = {}
    for start in range(0, TEST_COUNT, 128):
        worlds = worlds_all[start:start + 128]
        _, models, initial = build.e020.make_models(worlds)
        params = build.e020.parameters(worlds)
        for name, target in build.probes.library().items():
            final, trace = build.batch_probe_with_cost(
                models, initial, params, build.jp.asarray(target))
            jax.block_until_ready(trace)
            _, _, _, path, work, peak = final
            values = {
                f"{name}__trace": np.asarray(trace, dtype=np.float32),
                f"{name}__path_displacement": np.asarray(path, dtype=np.float32),
                f"{name}__absolute_work": np.asarray(work, dtype=np.float32),
                f"{name}__peak_speed": np.asarray(peak, dtype=np.float32),
            }
            for key, value in values.items():
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
    data = HERE / "data"
    audit = HERE / "audit"
    data.mkdir(exist_ok=True)
    audit.mkdir(exist_ok=True)
    input_path = data / "test_inputs.npz"
    target_path = audit / "test_targets.npz"
    np.savez_compressed(input_path, **inputs)
    np.savez_compressed(target_path, **targets)
    manifest = {
        "status": "fresh_test_generated_before_prediction",
        "seed": TEST_SEED,
        "worlds": TEST_COUNT,
        "probes": list(PROBES),
        "elapsed_seconds": time.perf_counter() - started,
        "input_sha256": sha256(input_path),
        "target_sha256": sha256(target_path),
        "prerun_lock_sha256": sha256(lock),
    }
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
