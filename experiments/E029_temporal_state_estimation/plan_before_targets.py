#!/usr/bin/env python3
"""Plan from every frozen state estimate before test-target access."""
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

PROBE = "fixed_negative_first"
BUDGETS = (9, 33, 101, 401)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


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
    for budget in BUDGETS:
        actions[f"budget_{budget}"] = planning.planned_actions(
            PROBE, estimates[f"budget_{budget}__estimated_parameters"],
            inputs["goal"], inputs["world_seed"])
    output_path = HERE / "audit/actions_before_targets.npz"
    np.savez_compressed(output_path, **actions)
    action_lock = {"status": "all_actions_frozen_before_target_access",
        "estimate_artifact_sha256": sha256(estimate_path),
        "action_artifact_sha256": sha256(output_path),
        "test_targets_accessed": False}
    (HERE / "audit/action_lock.json").write_text(
        json.dumps(action_lock, indent=2) + "\n")
    print(json.dumps(action_lock, indent=2))


if __name__ == "__main__":
    main()
