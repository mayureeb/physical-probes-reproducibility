#!/usr/bin/env python3
"""Plan from both state estimates and freeze actions before target access."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E028 = HERE.parent / "E028_model_based_planning"
sys.path.insert(0, str(E028))
import plan_before_targets as planning  # noqa: E402


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    estimate_path = HERE / "audit/estimates_before_targets.npz"
    lock = json.loads((HERE / "audit/estimation_lock.json").read_text())
    if sha256(estimate_path) != lock["estimate_artifact_sha256"]:
        raise RuntimeError("estimate artifact hash mismatch")
    estimates = load(estimate_path)
    inputs = load(HERE / "data/test_inputs.npz")
    actions = {}
    for name in ("inherited", "direct"):
        actions[name] = planning.planned_actions(
            "fixed_negative_first", estimates[name], inputs["goal"], inputs["world_seed"])
    output_path = HERE / "audit/actions_before_targets.npz"
    np.savez_compressed(output_path, **actions)
    record = {"status": "both_action_vectors_frozen_before_target_access",
        "estimate_artifact_sha256": sha256(estimate_path),
        "action_artifact_sha256": sha256(output_path),
        "test_targets_accessed": False}
    (HERE / "audit/action_lock.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

