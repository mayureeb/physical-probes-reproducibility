#!/usr/bin/env python3
"""Download, verify, align, and window the locked E024 vector signals."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CATALOG = REPO / "datasets/tartandrive2/catalog.json"
BASE = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartandrive2"
EXCLUDED = "2023-11-14-14-24-21_gupta"
DT = 0.1
HISTORY = 20
HORIZON = 10
MAX_GAP = 0.25
STRIDE = 2
SIGNALS = {
    "controls": "control.npy",
    "gps_odom": "odometry.npy",
    "multisense_imu": "imu.npy",
    "shock_pos": "shock.npy",
    "steering_angle": "float.npy",
    "wheel_rpm": "encoder.npy",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_for(record: dict) -> str:
    if "slag_heap" in record["course"]:
        return "test_slag_heap"
    if "damp" in record["conditions"]:
        return "test_damp"
    value = int(hashlib.sha256(record["run"].encode()).hexdigest()[:8], 16)
    return "validation" if value % 5 == 0 else "train"


def objects_for(run: str) -> list[tuple[str, str]]:
    prefix = f"kitti/all_topics/{run}"
    output = []
    for modality, filename in SIGNALS.items():
        output.append((f"{modality}/{filename}", f"{prefix}/{modality}/{filename}"))
        output.append((f"{modality}/timestamps.txt", f"{prefix}/{modality}/timestamps.txt"))
    return output


def download_one(raw_root: Path, run: str, relative: str, remote: str) -> dict:
    target = raw_root / run / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        error = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{BASE}/{remote}", timeout=120) as source:
                    temporary = target.with_suffix(target.suffix + ".partial")
                    temporary.write_bytes(source.read())
                    temporary.replace(target)
                error = None
                break
            except Exception as caught:  # bounded retry is recorded by failure
                error = caught
                time.sleep(2 ** attempt)
        if error is not None:
            raise error
    return {"run": run, "path": relative, "remote": remote,
            "bytes": target.stat().st_size, "sha256": sha256(target)}


def clean_series(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    keep = np.r_[times[1:] != times[:-1], True]
    duplicates = int(np.sum(~keep))
    return times[keep], values[keep], duplicates


def interpolate(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim == 1:
        values = values[:, None]
    index = np.searchsorted(times, grid, side="right")
    left = np.clip(index - 1, 0, len(times) - 1)
    right = np.clip(index, 0, len(times) - 1)
    gap = times[right] - times[left]
    valid = (grid >= times[0]) & (grid <= times[-1]) & (gap <= MAX_GAP)
    output = np.column_stack([np.interp(grid, times, values[:, column])
                              for column in range(values.shape[1])])
    return output.astype(np.float32), valid


def prepare_run(raw_root: Path, record: dict) -> tuple[dict, dict]:
    run = record["run"]
    series, duplicate_counts = {}, {}
    for modality, filename in SIGNALS.items():
        folder = raw_root / run / modality
        times = np.loadtxt(folder / "timestamps.txt", dtype=np.float64)
        values = np.load(folder / filename, allow_pickle=False).astype(np.float64)
        times, values, duplicates = clean_series(times, values)
        if len(times) != len(values) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid {run}/{modality}")
        series[modality] = (times, values)
        duplicate_counts[modality] = duplicates
    start = max(item[0][0] for item in series.values())
    stop = min(item[0][-1] for item in series.values())
    grid = np.arange(np.ceil(start / DT) * DT, stop + DT * 0.25, DT)
    aligned, valid = {}, np.ones(len(grid), dtype=bool)
    for modality, (times, values) in series.items():
        aligned[modality], modality_valid = interpolate(times, values, grid)
        valid &= modality_valid
    odom = aligned["gps_odom"]
    if odom.shape[1] != 13 or aligned["multisense_imu"].shape[1] != 6:
        raise RuntimeError(f"unexpected state shape in {run}")
    state = odom[:, [7, 8, 12]]
    steering = aligned["steering_angle"][:, :1]
    feature = np.concatenate((state, aligned["controls"], steering,
                              aligned["multisense_imu"], aligned["wheel_rpm"],
                              aligned["shock_pos"]), axis=1).astype(np.float32)
    histories, currents, targets, times_out = [], [], [], []
    first = HISTORY - 1
    last = len(grid) - HORIZON
    for current_index in range(first, last, STRIDE):
        used = slice(current_index - HISTORY + 1, current_index + HORIZON + 1)
        if not valid[used].all():
            continue
        histories.append(feature[current_index - HISTORY + 1:current_index + 1])
        currents.append(feature[current_index, :6])
        targets.append(state[current_index + HORIZON])
        times_out.append(grid[current_index])
    output = {
        "history": (np.asarray(histories, dtype=np.float32) if histories else
                    np.empty((0, HISTORY, feature.shape[1]), dtype=np.float32)),
        "current": (np.asarray(currents, dtype=np.float32) if currents else
                    np.empty((0, 6), dtype=np.float32)),
        "target": (np.asarray(targets, dtype=np.float32) if targets else
                   np.empty((0, 3), dtype=np.float32)),
        "time": np.asarray(times_out, dtype=np.float64),
        "run": np.full(len(histories), run),
    }
    audit = {"run": run, "split": split_for(record), "windows": len(histories),
             "duplicates_removed": duplicate_counts, "grid_rows": len(grid),
             "feature_channels": feature.shape[1]}
    return output, audit


def concatenate(parts: list[dict]) -> dict:
    return {key: np.concatenate([part[key] for part in parts], axis=0)
            for key in parts[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "data")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text())
    eligible = [record for record in catalog["runs"]
                if record["kitti_available"] and record["run"] != EXCLUDED]
    raw_root = args.root / "raw"
    jobs = [(raw_root, record["run"], relative, remote)
            for record in eligible for relative, remote in objects_for(record["run"])]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        downloaded = list(pool.map(lambda job: download_one(*job), jobs))
    (args.root / "download_manifest.json").write_text(json.dumps({
        "status": "e024_primary_download", "base": BASE,
        "catalog_sha256": sha256(CATALOG), "files": downloaded,
    }, indent=2) + "\n")
    grouped: dict[str, list[dict]] = {name: [] for name in
        ("train", "validation", "test_slag_heap", "test_damp")}
    audits = []
    for index, record in enumerate(eligible, 1):
        part, audit = prepare_run(raw_root, record)
        grouped[audit["split"]].append(part)
        audits.append(audit)
        print(json.dumps({"prepared": index, "count": len(eligible), **audit}), flush=True)
    split_manifest = {}
    for name, parts in grouped.items():
        if not parts:
            raise RuntimeError(f"empty split: {name}")
        combined = concatenate(parts)
        if name.startswith("test_"):
            path = args.root / f"{name}_inputs.npz"
            np.savez_compressed(path, **{key: value for key, value in combined.items()
                                        if key != "target"})
            target_path = HERE / "audit" / f"{name}_targets.npz"
            target_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(target_path, target=combined["target"],
                                run=combined["run"], time=combined["time"])
            target_sha256 = sha256(target_path)
        else:
            path = args.root / f"{name}.npz"
            np.savez_compressed(path, **combined)
            target_sha256 = None
        split_manifest[name] = {"runs": sorted(set(combined["run"].tolist())),
                                "windows": len(combined["target"]),
                                "input_sha256": sha256(path),
                                "evaluator_target_sha256": target_sha256}
    (args.root / "manifest.json").write_text(json.dumps({
        "status": "e024_prepared_before_modeling", "dt_seconds": DT,
        "history_steps": HISTORY, "horizon_steps": HORIZON,
        "maximum_interpolation_gap_seconds": MAX_GAP, "stride_steps": STRIDE,
        "feature_order": ["vx", "vy", "yaw_rate", "control_0", "control_1", "steering",
                          "imu_wx", "imu_wy", "imu_wz", "imu_ax", "imu_ay", "imu_az",
                          "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr",
                          "shock_fl", "shock_fr", "shock_rl", "shock_rr"],
        "target_order": ["future_vx", "future_vy", "future_yaw_rate"],
        "splits": split_manifest, "run_audit": audits,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
