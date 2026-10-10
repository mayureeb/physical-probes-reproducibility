#!/usr/bin/env python3
"""Generate fresh selected-probe data while keeping test targets separate."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
sys.path.insert(0, str(SOURCE))
import build_data as build  # noqa: E402

SPLITS = {"train": (30130, 4096), "validation": (30230, 1024),
          "test": (30330, 2048)}
CHUNK_SIZE = 256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_waveforms() -> dict[str, np.ndarray]:
    with np.load(HERE / "audit/selected_waveforms.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    selection_path = HERE / "audit/selection_lock.json"
    source_lock = HERE / "audit/confirmatory_source_lock.json"
    runtime_lock = HERE / "audit/amendment_003_lock.json"
    if not selection_path.exists() or not source_lock.exists() or not runtime_lock.exists():
        raise RuntimeError("selection, source, and runtime locks are required")
    selection = json.loads(selection_path.read_text())
    if selection["confirmatory_data_generated"] or selection["test_targets_accessed"]:
        raise RuntimeError("selection lock does not preserve the required boundary")
    waveforms = load_waveforms()
    original_library = build.probes.library
    build.probes.library = lambda: waveforms
    manifest = {"status": "generated_before_model_selection", "splits": {},
                "selection_lock_sha256": sha256(selection_path),
                "confirmatory_source_lock_sha256": sha256(source_lock),
                "runtime_amendment_sha256": sha256(runtime_lock),
                "chunk_size": CHUNK_SIZE}
    try:
        for split, (seed, count) in SPLITS.items():
            started = time.perf_counter()
            arrays = build.generate_split(seed, count, CHUNK_SIZE)
            if split == "test":
                target_suffixes = ("__interval_low", "__interval_high", "__oracle_speed",
                                   "__oracle_error", "__target_attained", "__stopped")
                target_keys = {key for key in arrays if key.endswith(target_suffixes)}
                target_keys.add("parameters")
                inputs = {key: value for key, value in arrays.items() if key not in target_keys}
                targets = {key: value for key, value in arrays.items()
                           if key in target_keys or key in ("goal", "world_seed")}
                path = HERE / "data/test_inputs.npz"
                target_path = HERE / "audit/test_targets.npz"
                np.savez_compressed(path, **inputs)
                np.savez_compressed(target_path, **targets)
                target_hash = sha256(target_path)
            else:
                path = HERE / f"data/{split}.npz"
                np.savez_compressed(path, **arrays)
                target_hash = None
            manifest["splits"][split] = {"seed": seed, "worlds": count,
                "input_sha256": sha256(path), "target_sha256": target_hash,
                "elapsed_seconds": time.perf_counter() - started}
    finally:
        build.probes.library = original_library
    output = HERE / "data/manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
