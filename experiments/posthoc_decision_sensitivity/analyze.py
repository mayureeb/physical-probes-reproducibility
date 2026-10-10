#!/usr/bin/env python3
"""Post hoc sensitivity of successful action intervals to physical parameters."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E025 = ROOT / "experiments/E025_decision_ambiguity"
E028 = ROOT / "experiments/E028_model_based_planning"
sys.path.insert(0, str(E025))
sys.path.insert(0, str(E028))
import build_data as build  # noqa: E402
import plan_before_targets as planning  # noqa: E402

PARAMETERS = ("mass", "mu", "damping", "gain", "strength", "tau1", "tau2", "mix")
LABELS = (
    "Mass", "Coulomb friction", "Damping", "Actuator gain",
    "Resistance strength", "Short time constant", "Long time constant", "Mixture weight",
)
LOW = np.asarray((0.6, 0.06, 0.01, 14.0, 0.2, 0.05, 0.7, 0.25), dtype=np.float32)
HIGH = np.asarray((1.5, 0.22, 0.12, 30.0, 1.0, 0.25, 1.8, 0.7), dtype=np.float32)
PROBE = "fixed_negative_first"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def landing_at_speeds(parameters: np.ndarray, goals: np.ndarray, seeds: np.ndarray,
                      speeds: list[np.ndarray], batch_size: int) -> list[np.ndarray]:
    outputs = [[] for _ in speeds]
    waveform = build.probes.library()[PROBE]
    for start in range(0, len(goals), batch_size):
        stop = min(start + batch_size, len(goals))
        worlds = planning.worlds(parameters[start:stop], goals[start:stop], seeds[start:stop])
        _, models, initial = build.e020.make_models(worlds)
        physical = build.e020.parameters(worlds)
        final, _ = build.batch_probe_with_cost(
            models, initial, physical, build.jp.asarray(waveform))
        data, z1, z2, *_ = final
        for output, requested in zip(outputs, speeds):
            result = build.e021.batch_release(
                models, data, physical, z1, z2,
                build.jp.asarray(requested[start:stop], dtype=build.jp.float32))
            jax.block_until_ready(result[0])
            output.append(np.asarray(result[0], dtype=np.float32))
    return [np.concatenate(parts) for parts in outputs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets", type=Path,
        default=ROOT / "experiments/E030_decision_directed_probe/audit/test_targets.npz")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--subset-seed", type=int, default=31032)
    parser.add_argument("--step-fraction", type=float, default=0.02)
    parser.add_argument("--speed-step", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--parameter-index", type=int, choices=range(8))
    args = parser.parse_args()

    source = load(args.targets)
    all_theta = source["parameters"].astype(np.float32)
    scale = all_theta.std(axis=0)
    rng = np.random.default_rng(args.subset_seed)
    selected = np.sort(rng.choice(len(all_theta), size=args.worlds, replace=False))
    theta = all_theta[selected]
    goals = source["goal"].astype(np.float32)[selected]
    seeds = source["world_seed"][selected]
    baseline_low = source[f"{PROBE}__interval_low"].astype(np.float32)[selected]
    baseline_high = source[f"{PROBE}__interval_high"].astype(np.float32)[selected]
    step = args.step_fraction * (HIGH - LOW)
    started = time.perf_counter()

    low_minus = np.maximum(0.0, baseline_low - args.speed_step)
    low_plus = np.minimum(3.0, baseline_low + args.speed_step)
    high_minus = np.maximum(0.0, baseline_high - args.speed_step)
    high_plus = np.minimum(3.0, baseline_high + args.speed_step)
    base_distances = landing_at_speeds(
        theta, goals, seeds, [low_minus, low_plus, high_minus, high_plus], args.batch_size)
    low_speed_derivative = ((base_distances[1] - base_distances[0]) /
                            (low_plus - low_minus))
    high_speed_derivative = ((base_distances[3] - base_distances[2]) /
                             (high_plus - high_minus))
    if np.any(low_speed_derivative <= 0) or np.any(high_speed_derivative <= 0):
        raise RuntimeError("positive local landing-distance derivatives are required")
    rows = []
    parameter_indices = (range(8) if args.parameter_index is None
                         else (args.parameter_index,))
    for index in parameter_indices:
        name, label = PARAMETERS[index], LABELS[index]
        minus, plus = theta.copy(), theta.copy()
        minus[:, index] = np.maximum(LOW[index], theta[:, index] - step[index])
        plus[:, index] = np.minimum(HIGH[index], theta[:, index] + step[index])
        minus_low_distance, minus_high_distance = landing_at_speeds(
            minus, goals, seeds, [baseline_low, baseline_high], args.batch_size)
        plus_low_distance, plus_high_distance = landing_at_speeds(
            plus, goals, seeds, [baseline_low, baseline_high], args.batch_size)
        denominator = plus[:, index] - minus[:, index]
        low_distance_derivative = (plus_low_distance - minus_low_distance) / denominator
        high_distance_derivative = (plus_high_distance - minus_high_distance) / denominator
        low_derivative = -low_distance_derivative / low_speed_derivative
        high_derivative = -high_distance_derivative / high_speed_derivative
        # For intervals on the real line, the Hausdorff distance is the larger
        # absolute shift of their two boundaries.
        sensitivity = np.maximum(np.abs(low_derivative), np.abs(high_derivative)) * scale[index]
        midpoint_derivative = 0.5 * (low_derivative + high_derivative) * scale[index]
        rows.append({
            "parameter": name,
            "label": label,
            "test_standard_deviation": float(scale[index]),
            "finite_difference_step": float(step[index]),
            "mean_interval_boundary_sensitivity_mps_per_sd": float(np.mean(sensitivity)),
            "median_interval_boundary_sensitivity_mps_per_sd": float(np.median(sensitivity)),
            "mean_absolute_midpoint_sensitivity_mps_per_sd": float(np.mean(np.abs(midpoint_derivative))),
        })
        print(json.dumps(rows[-1]), flush=True)

    result = {
        "status": "post_hoc_on_locked_test_worlds",
        "worlds": int(len(theta)),
        "source_worlds": int(len(all_theta)),
        "subset_seed": args.subset_seed,
        "selected_index_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "probe": PROBE,
        "parameter_order": list(PARAMETERS),
        "target_sha256": sha256(args.targets),
        "step_fraction_of_parameter_range": args.step_fraction,
        "release_speed_finite_difference_step_mps": args.speed_step,
        "metric": (
            "Mean local Hausdorff sensitivity of the successful release-speed interval "
            "from implicit boundary derivatives, in m/s per test-set parameter standard deviation."
        ),
        "baseline_interval_width_mps": {
            "mean": float(np.mean(baseline_high - baseline_low)),
            "median": float(np.median(baseline_high - baseline_low)),
        },
        "per_parameter": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "devices": [str(device) for device in jax.devices()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
