#!/usr/bin/env python3
"""Replay post hoc structured-decoder actions in the registered simulator."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
TOLERANCES_M = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sys.path.insert(0, str(SOURCE))
    import evaluate

    prediction_path = HERE / "structured_decoder_predictions.npz"
    target_path = SOURCE / "data/test_inputs.npz"
    selection_path = SOURCE / "audit/selection.json"
    predictions = load(prediction_path)
    inputs = load(target_path)
    selection = json.loads(selection_path.read_text())
    physical = evaluate.physical_predictions(predictions, selection)
    arrays, methods = {}, {}
    for label, values in physical.items():
        error = np.abs(values["distance"] - inputs["goal"])
        arrays[f"{label}__landing_error"] = error.astype(np.float32)
        methods[label] = {
            "mean_landing_error_m": float(error.mean()),
            "completion_by_tolerance_m": {
                str(tolerance): float(np.mean(error <= tolerance))
                for tolerance in TOLERANCES_M
            },
        }
    payload = {
        "status": "post_hoc_structured_physical_evaluation_complete",
        "source_sha256": {
            "structured_decoder_predictions.npz": sha256(prediction_path),
            "test_inputs.npz": sha256(target_path),
            "selection.json": sha256(selection_path),
        },
        "methods": methods,
    }
    (HERE / "structured_physical_results.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(HERE / "structured_physical_predictions.npz", **arrays)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
