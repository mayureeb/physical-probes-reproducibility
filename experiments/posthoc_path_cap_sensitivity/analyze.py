#!/usr/bin/env python3
"""Recompute fixed-sweep completion after excluding path-cap exceedances."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E030_decision_directed_probe"
PROBES = ("fixed_positive_first", "fixed_negative_first")
METHODS = ("direct", "estimated_planner")


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def main() -> None:
    inputs = load(SOURCE / "data/test_inputs.npz")
    targets = load(SOURCE / "audit/test_targets.npz")
    actions = load(SOURCE / "audit/actions_before_targets.npz")
    recorded = json.loads((SOURCE / "results.json").read_text())
    caps = json.loads((SOURCE / "audit/selection_lock.json").read_text())["caps"]
    result = {"status": "posthoc_path_cap_sensitivity",
              "path_cap": caps["path_displacement"], "probes": {}}
    for probe in PROBES:
        path = inputs[f"{probe}__path_displacement"]
        eligible = path <= caps["path_displacement"] + 1e-6
        probe_result = {"worlds": int(len(path)), "cap_compliant_worlds": int(eligible.sum()),
                        "excluded_worlds": int((~eligible).sum()), "methods": {}}
        low = targets[f"{probe}__interval_low"]
        high = targets[f"{probe}__interval_high"]
        for method in METHODS:
            action = actions[f"{probe}__{method}"]
            success = (action >= low) & (action <= high)
            physical = recorded["descriptive"][f"{probe}__{method}"]["completion"]
            probe_result["methods"][method] = {
                "interval_completion_all": float(success.mean()),
                "recorded_physical_completion_all": float(physical),
                "absolute_reproduction_difference": float(abs(success.mean() - physical)),
                "cap_compliant_completion": float(success[eligible].mean()),
                "excluded_world_completion": float(success[~eligible].mean())}
        result["probes"][probe] = probe_result
    HERE.mkdir(exist_ok=True)
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
