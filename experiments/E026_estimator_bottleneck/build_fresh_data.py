#!/usr/bin/env python3
"""Generate the fresh E026 two-probe splits using the locked E025 simulator."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E025 = HERE.parent / "E025_decision_ambiguity"
sys.path.insert(0, str(E025))
import build_data as build  # noqa: E402

SPLITS = {"train": (26026, 4096), "validation": (26126, 512), "test": (26226, 512)}
PROBES = ("fixed_positive_first", "multisine")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data_dir = HERE / "data"
    audit_dir = HERE / "audit"
    data_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    complete_library = build.probes.library()
    build.probes.library = lambda: {name: complete_library[name] for name in PROBES}
    manifest = {"status": "fresh_data_generated_before_selection", "splits": {},
                "probes": list(PROBES), "source": "E025 locked simulator and probes"}
    for split, (seed, count) in SPLITS.items():
        started = time.perf_counter()
        arrays = build.generate_split(seed, count, 128)
        if split == "test":
            suffixes = ("__interval_low", "__interval_high", "__oracle_speed",
                        "__oracle_error", "__target_attained", "__stopped")
            target_keys = {key for key in arrays if key.endswith(suffixes)} | {"parameters"}
            inputs = {key: value for key, value in arrays.items() if key not in target_keys}
            targets = {key: value for key, value in arrays.items()
                       if key in target_keys or key in ("goal", "world_seed")}
            path = data_dir / "test_inputs.npz"
            target_path = audit_dir / "test_targets.npz"
            np.savez_compressed(path, **inputs)
            np.savez_compressed(target_path, **targets)
            target_hash = digest(target_path)
        else:
            path = data_dir / f"{split}.npz"
            np.savez_compressed(path, **arrays)
            target_hash = None
        manifest["splits"][split] = {"seed": seed, "rows": count,
            "seconds": time.perf_counter() - started, "sha256": digest(path),
            "target_sha256": target_hash}
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
