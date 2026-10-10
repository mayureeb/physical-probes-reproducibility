#!/usr/bin/env python3
"""Generate design worlds and freeze task-directed and D-optimal selections."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
sys.path.insert(0, str(SOURCE))
import build_data as build  # noqa: E402
import probe_candidates  # noqa: E402

DESIGN_SEED = 30030
DESIGN_WORLDS = 512
TRACE_INDEX = np.arange(0, 401, 50)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed(trace: np.ndarray, name: str) -> np.ndarray:
    value = trace[:, TRACE_INDEX].astype(np.float32).copy()
    seed = int(hashlib.sha256(f"E030:design:{name}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    value[:, :, 0] += rng.normal(0, 0.01, value[:, :, 0].shape)
    innovation = rng.normal(0, 0.04, value[:, :, 1].shape).astype(np.float32)
    noise = np.empty_like(innovation)
    noise[:, 0] = innovation[:, 0]
    factor = np.sqrt(1 - 0.8 ** 2)
    for index in range(1, noise.shape[1]):
        noise[:, index] = 0.8 * noise[:, index - 1] + factor * innovation[:, index]
    value[:, :, 1] += noise
    return value


def cv_score(data: dict[str, np.ndarray], name: str) -> dict[str, float]:
    trace = observed(data[f"{name}__trace"], name)
    x = np.concatenate((trace.reshape(len(trace), -1), data["goal"][:, None]), axis=1)
    y = data[f"{name}__oracle_speed"]
    low, high = data[f"{name}__interval_low"], data[f"{name}__interval_high"]
    prediction = np.empty_like(y)
    folds = KFold(n_splits=5, shuffle=True, random_state=30030)
    for fold, (train, test) in enumerate(folds.split(x)):
        model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=4,
            max_features=1.0, n_jobs=-1, random_state=30030 + fold)
        model.fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    return {
        "completion": float(np.mean((prediction >= low) & (prediction <= high))),
        "speed_mae": float(np.mean(np.abs(prediction - y))),
    }


def d_optimal_score(trace: np.ndarray, parameters: np.ndarray) -> float:
    """Inherited standardized local-sensitivity log determinant."""
    x = trace.reshape(len(trace), -1).astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    p = (parameters - parameters.mean(0)) / (parameters.std(0) + 1e-6)
    coefficient = np.linalg.lstsq(p, x, rcond=None)[0]
    sign, value = np.linalg.slogdet(
        coefficient @ coefficient.T + 1e-6 * np.eye(p.shape[1]))
    return float(value if sign > 0 else -np.inf)


def main() -> None:
    protocol = HERE / "PROTOCOL.md"
    source_lock = HERE / "audit/prerun_lock.json"
    if not source_lock.exists():
        raise RuntimeError("audit/prerun_lock.json must be committed before generation")
    candidates = probe_candidates.library()
    original_library = build.probes.library
    build.probes.library = lambda: candidates
    try:
        data = build.generate_split(DESIGN_SEED, DESIGN_WORLDS, 32)
    finally:
        build.probes.library = original_library
    data_path = HERE / "data/design.npz"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, **data)

    fixed = "fixed_positive_first"
    caps = {
        "path_displacement": float(data[f"{fixed}__path_displacement"].max()),
        "absolute_work": float(data[f"{fixed}__absolute_work"].max()),
    }
    records, eligible = {}, []
    for name in candidates:
        path = data[f"{name}__path_displacement"]
        work = data[f"{name}__absolute_work"]
        violations = {
            "path": int(np.sum(path > caps["path_displacement"] + 1e-6)),
            "work": int(np.sum(work > caps["absolute_work"] + 1e-6)),
        }
        record = {"cost_violations": violations}
        if not violations["path"] and not violations["work"]:
            eligible.append(name)
            record["decision_score"] = cv_score(data, name)
            record["d_optimal_score"] = d_optimal_score(
                data[f"{name}__trace"][:, TRACE_INDEX], data["parameters"])
        records[name] = record
    if not eligible:
        raise RuntimeError("no candidate met the locked physical caps")
    task_probe = min(eligible, key=lambda name: (
        -records[name]["decision_score"]["completion"],
        records[name]["decision_score"]["speed_mae"], name))
    d_probe = max(eligible, key=lambda name: (records[name]["d_optimal_score"], name))
    waveforms_path = HERE / "audit/selected_waveforms.npz"
    np.savez_compressed(waveforms_path,
        decision_directed=candidates[task_probe], d_optimal=candidates[d_probe],
        fixed_positive_first=candidates["fixed_positive_first"],
        fixed_negative_first=candidates["fixed_negative_first"])
    selection = {
        "status": "candidate_selection_frozen_before_confirmatory_generation",
        "protocol_sha256": sha256(protocol),
        "prerun_lock_sha256": sha256(source_lock),
        "design_seed": DESIGN_SEED,
        "design_worlds": DESIGN_WORLDS,
        "design_data_sha256": sha256(data_path),
        "selected_waveforms_sha256": sha256(waveforms_path),
        "caps": caps,
        "eligible_candidates": eligible,
        "decision_directed_source": task_probe,
        "d_optimal_source": d_probe,
        "records": records,
        "confirmatory_data_generated": False,
        "test_targets_accessed": False,
    }
    output = HERE / "audit/selection_lock.json"
    output.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps({"decision_directed": task_probe, "d_optimal": d_probe,
                      "eligible": len(eligible)}, indent=2))


if __name__ == "__main__":
    main()
