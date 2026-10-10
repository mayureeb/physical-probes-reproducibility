#!/usr/bin/env python3
"""Fit the simulator to fresh traces without loading evaluator parameters."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
E029 = HERE.parent / "E029_temporal_state_estimation"
sys.path.insert(0, str(E029))
import fit_estimators as temporal  # noqa: E402
import prototype_fit as direct  # noqa: E402

BATCH = 64
RESTARTS = 2
STEPS = 50


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed(trace: np.ndarray) -> np.ndarray:
    values = trace.astype(np.float32).copy()
    seed = int(hashlib.sha256(b"E031:test:noise").hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    values[:, :, 0] += rng.normal(0, temporal.VELOCITY_SIGMA,
                                  values[:, :, 0].shape).astype(np.float32)
    innovation = rng.normal(0, temporal.FORCE_SIGMA,
                            values[:, :, 1].shape).astype(np.float32)
    noise = np.empty_like(innovation)
    noise[:, 0] = innovation[:, 0]
    factor = np.sqrt(1.0 - temporal.STEP_RHO ** 2)
    for index in range(1, noise.shape[1]):
        noise[:, index] = temporal.STEP_RHO * noise[:, index - 1] + factor * innovation[:, index]
    values[:, :, 1] += noise
    return values[:, direct.TRACE_INDEX]


def main() -> None:
    data = load(HERE / "data/test_inputs.npz")
    trace = observed(data["trace"])
    inherited_model = joblib.load(E029 / "checkpoints/budget_9_estimator.joblib")
    initial = inherited_model.predict(temporal.features(trace, 9))
    inherited_train = load(E029 / "data/train.npz")["parameters"]
    inherited_validation = load(E029 / "data/validation.npz")["parameters"]
    inherited_reference = np.concatenate((inherited_train, inherited_validation))
    initial = np.clip(initial, inherited_reference.min(0),
                      inherited_reference.max(0)).astype(np.float32)
    estimates, losses = [], []
    histories = []
    started = time.perf_counter()
    for start in range(0, len(trace), BATCH):
        end = min(start + BATCH, len(trace))
        estimate, loss, history = direct.fit(
            trace[start:end], initial[start:end], RESTARTS, STEPS, 3133101 + start)
        estimates.append(estimate.astype(np.float32))
        losses.append(loss.astype(np.float32))
        histories.append({"start": start, "end": end, "history": history})
        print(json.dumps({"fitted": end, "count": len(trace),
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    output_path = HERE / "audit/estimates_before_targets.npz"
    np.savez_compressed(output_path, inherited=initial,
                        direct=np.concatenate(estimates),
                        direct_whitened_trace_loss=np.concatenate(losses))
    record = {"status": "state_estimates_frozen_before_target_access",
        "test_input_sha256": sha256(HERE / "data/test_inputs.npz"),
        "estimate_artifact_sha256": sha256(output_path),
        "inherited_checkpoint_sha256": sha256(E029 / "checkpoints/budget_9_estimator.joblib"),
        "batch": BATCH, "restarts": RESTARTS, "steps": STEPS,
        "test_targets_accessed": False, "elapsed_seconds": time.perf_counter() - started,
        "batch_histories": histories}
    (HERE / "audit/estimation_lock.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
