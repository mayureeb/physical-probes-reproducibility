#!/usr/bin/env python3
"""Generate full traces and separate test physical parameters."""
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

SPLITS = {"train": (29029, 4096), "validation": (29129, 1024),
          "test": (29229, 2048)}
PROBE = "fixed_negative_first"
BATCH = 128


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(seed: int, count: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    worlds_all = build.make_worlds(seed, count)
    target = build.probes.library()[PROBE]
    traces = []
    started = time.perf_counter()
    for start in range(0, count, BATCH):
        worlds = worlds_all[start:start + BATCH]
        _, models, initial = build.e020.make_models(worlds)
        params = build.e020.parameters(worlds)
        _, trace = build.batch_probe_with_cost(models, initial, params,
                                                build.jp.asarray(target))
        jax.block_until_ready(trace)
        traces.append(np.asarray(trace, dtype=np.float32))
        print(json.dumps({"split_seed": seed, "built": start + len(worlds),
                          "count": count,
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    inputs = {
        "trace": np.concatenate(traces),
        "goal": np.asarray([world.goal for world in worlds_all], dtype=np.float32),
        "world_seed": np.asarray([world.seed for world in worlds_all], dtype=np.int64),
    }
    parameters = np.asarray([[getattr(world, key) for key in build.PARAMETERS]
                             for world in worlds_all], dtype=np.float32)
    return inputs, parameters


def main() -> None:
    lock = HERE / "audit/prerun_lock.json"
    if not lock.exists():
        raise RuntimeError("prerun lock is required")
    manifest = {"status": "generated_before_model_selection", "splits": {}}
    for split, (seed, count) in SPLITS.items():
        inputs, parameters = generate(seed, count)
        if split == "test":
            input_path = HERE / "data/test_inputs.npz"
            target_path = HERE / "audit/test_targets.npz"
            np.savez_compressed(input_path, **inputs)
            np.savez_compressed(target_path, parameters=parameters,
                                goal=inputs["goal"], world_seed=inputs["world_seed"])
            target_hash = sha256(target_path)
        else:
            input_path = HERE / f"data/{split}.npz"
            np.savez_compressed(input_path, **inputs, parameters=parameters)
            target_hash = None
        manifest["splits"][split] = {"seed": seed, "worlds": count,
            "input_sha256": sha256(input_path), "target_sha256": target_hash}
    manifest["prerun_lock_sha256"] = sha256(lock)
    (HERE / "data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
