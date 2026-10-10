#!/usr/bin/env python3
"""Quantify same-goal decision equivalence in the held-out matched-probe worlds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCES = {
    "matched_probe_test": HERE.parent / "E025_decision_ambiguity/audit/test_targets.npz",
    "probe_selection_test": HERE.parent / "E030_decision_directed_probe/audit/test_targets.npz",
}
PARAMETERS = ("mass", "mu", "damping", "gain", "strength", "tau1", "tau2", "mix")


def summarize(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
            "maximum": float(values.max())}


def analyze(source: Path) -> dict:
    with np.load(source, allow_pickle=False) as handle:
        data = {key: handle[key] for key in handle.files}
    probes = sorted(key.removesuffix("__interval_low") for key in data
                    if key.endswith("__interval_low"))
    scale = data["parameters"].std(0) + 1e-8
    records = {}
    for probe in probes:
        equivalent, distance = [], []
        low = data[f"{probe}__interval_low"]
        high = data[f"{probe}__interval_high"]
        for goal in np.unique(data["goal"]):
            indices = np.flatnonzero(data["goal"] == goal)
            for offset, first in enumerate(indices[:-1]):
                second = indices[offset + 1:]
                overlap = np.minimum(high[first], high[second]) - np.maximum(low[first], low[second])
                equivalent.extend((overlap >= 0).tolist())
                delta = (data["parameters"][second] - data["parameters"][first]) / scale
                distance.extend(np.sqrt(np.sum(delta ** 2, axis=1)).tolist())
        equivalent = np.asarray(equivalent, dtype=bool)
        distance = np.asarray(distance)
        records[probe] = {"same_goal_pairs": int(len(equivalent)),
            "decision_equivalent_pairs": int(equivalent.sum()),
            "decision_equivalent_fraction": float(equivalent.mean()),
            "decision_conflicting_fraction": float((~equivalent).mean()),
            "standardized_parameter_distance_equivalent": summarize(distance[equivalent]),
            "standardized_parameter_distance_conflicting": summarize(distance[~equivalent])}
    return {"source": str(source.relative_to(HERE.parents[1])),
        "worlds": int(len(data["goal"])), "goals": int(len(np.unique(data["goal"]))),
        "parameter_order": PARAMETERS, "probes": records}


def main() -> None:
    result = {"status": "posthoc_decision_equivalence_audit",
              "studies": {name: analyze(path) for name, path in SOURCES.items()}}
    HERE.mkdir(exist_ok=True)
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
