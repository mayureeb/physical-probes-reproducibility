#!/usr/bin/env python3
"""Generate fresh E025 matched-probe traces and feasible decision intervals."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

HERE = Path(__file__).resolve().parent
SPLITS = {"calibration": (25025, 512), "validation": (25125, 128),
          "test": (25225, 256)}
PARAMETERS = ("mass", "mu", "damping", "gain", "strength", "tau1", "tau2", "mix")
BISECTION_STEPS = 14
SMOKE_SEED = 25000
GOAL_GRID = np.linspace(0.3, 1.0, 32, dtype=np.float64)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


e020 = load_module("e025_e020", HERE.parent / "E020_mjx_full_sequence" / "run.py")
e021 = load_module("e025_e021", HERE.parent / "E021_mjx_full_sequence_acceptance" / "run.py")
probes = load_module("e025_probes", HERE / "probe_library.py")


def one_probe_with_cost(model, data, material, targets):
    gain, strength, tau1, tau2, mix = material

    def step(carry, target):
        current, z1, z2, path, work, peak = carry
        force, next_z1, next_z2 = e020.applied_force(
            current, strength, tau1, tau2, mix, z1, z2)
        control = jp.clip(gain * (target - current.qvel[0]), -50, 50)
        advanced = mjx.step(model, current.replace(
            qfrc_applied=force, ctrl=current.ctrl.at[0].set(control)))
        path = path + jp.abs(advanced.qpos[0] - current.qpos[0])
        work = work + jp.abs(control * advanced.qvel[0]) * e020.e014.DT
        peak = jp.maximum(peak, jp.abs(advanced.qvel[0]))
        observed = jp.stack((advanced.qvel[0],
                             advanced.ctrl[0] + advanced.qfrc_constraint[0]))
        return (advanced, next_z1, next_z2, path, work, peak), observed

    initial = (data, jp.float32(0), jp.float32(0), jp.float32(0),
               jp.float32(0), jp.float32(0))
    return jax.lax.scan(step, initial, targets)


@jax.jit
def batch_probe_with_cost(models, data, params, targets):
    material = (params["gain"], params["strength"], params["tau1"],
                params["tau2"], params["mix"])
    return jax.vmap(one_probe_with_cost, in_axes=(0, 0, 0, None))(
        models, data, material, targets)


def boundary(models, data, params, z1, z2, target):
    low, high = jp.zeros_like(target), jp.full_like(target, 3.0)
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2
        distance, *_ = e021.batch_release(models, data, params, z1, z2, middle)
        low = jp.where(distance < target, middle, low)
        high = jp.where(distance < target, high, middle)
    return (low + high) / 2


def concatenate_trees(trees):
    return jax.tree.map(lambda *leaves: jp.concatenate(leaves, axis=0), *trees)


def repeat_tree(tree, repeats: int):
    return jax.tree.map(lambda leaf: jp.concatenate([leaf] * repeats, axis=0), tree)


def repeat_params(params: dict, repeats: int) -> dict:
    return {key: jp.concatenate([value] * repeats, axis=0)
            for key, value in params.items()}


def make_worlds(seed: int, count: int):
    worlds = e020.e014.make_worlds(seed, count, [4])
    goals = np.resize(GOAL_GRID, count).copy()
    np.random.default_rng(seed + 777).shuffle(goals)
    return [dataclasses.replace(world, goal=float(goal))
            for world, goal in zip(worlds, goals)]


def smoke_monotonicity() -> dict:
    worlds = e020.e014.make_worlds(SMOKE_SEED, 8, [4])
    _, models, data = e020.make_models(worlds)
    params = e020.parameters(worlds)
    report = {}
    grid = np.linspace(0.0, 3.0, 31, dtype=np.float32)
    library = probes.library()
    final_states = []
    for target in library.values():
        final, _ = batch_probe_with_cost(models, data, params, jp.asarray(target))
        final_states.append(final)
    count_probes, count_worlds = len(library), len(worlds)
    flat_data = concatenate_trees([final[0] for final in final_states])
    flat_z1 = jp.concatenate([final[1] for final in final_states])
    flat_z2 = jp.concatenate([final[2] for final in final_states])
    flat_models = repeat_tree(models, count_probes)
    flat_params = repeat_params(params, count_probes)
    distances = []
    for speed in grid:
        request = jp.full((count_probes * count_worlds,), speed, dtype=jp.float32)
        outcome = e021.batch_release(flat_models, flat_data, flat_params,
                                     flat_z1, flat_z2, request)
        distances.append(np.asarray(outcome[0]).reshape(count_probes, count_worlds))
    matrices = np.stack(distances, axis=2)
    for probe_index, name in enumerate(library):
        matrix = matrices[probe_index]
        minimum_step = np.min(np.diff(matrix, axis=1), axis=1)
        report[name] = {"worlds": len(worlds), "grid_speeds": len(grid),
                        "monotone_worlds": int(np.sum(minimum_step >= -1e-4)),
                        "minimum_distance_step_m": float(np.min(minimum_step))}
    report["passed"] = all(item["monotone_worlds"] == item["worlds"]
                           for item in report.values() if isinstance(item, dict))
    return report


def generate_split(seed: int, count: int, chunk_size: int) -> dict[str, np.ndarray]:
    worlds_all = make_worlds(seed, count)
    waveforms = probes.library()
    collected: dict[str, list[np.ndarray]] = {}
    started = time.perf_counter()
    for start in range(0, count, chunk_size):
        worlds = worlds_all[start:start + chunk_size]
        _, models, initial = e020.make_models(worlds)
        params = e020.parameters(worlds)
        goals = jp.asarray([world.goal for world in worlds], dtype=jp.float32)
        names, final_states, traces = [], [], []
        for name, target in waveforms.items():
            final, trace = batch_probe_with_cost(models, initial, params, jp.asarray(target))
            names.append(name)
            final_states.append(final)
            traces.append(trace)
        count_probes, count_worlds = len(names), len(worlds)
        flat_data = concatenate_trees([final[0] for final in final_states])
        flat_z1 = jp.concatenate([final[1] for final in final_states])
        flat_z2 = jp.concatenate([final[2] for final in final_states])
        flat_models = repeat_tree(models, count_probes)
        flat_params = repeat_params(params, count_probes)
        flat_goals = jp.concatenate([goals] * count_probes)
        paired_models = repeat_tree(flat_models, 2)
        paired_data = repeat_tree(flat_data, 2)
        paired_params = repeat_params(flat_params, 2)
        paired_z1 = jp.concatenate([flat_z1, flat_z1])
        paired_z2 = jp.concatenate([flat_z2, flat_z2])
        paired_targets = jp.concatenate((jp.maximum(0.0, flat_goals - e020.e014.TOLERANCE),
                                         flat_goals + e020.e014.TOLERANCE))
        paired_boundaries = boundary(paired_models, paired_data, paired_params,
                                     paired_z1, paired_z2, paired_targets)
        flat_lower, flat_upper = jp.split(paired_boundaries, 2)
        flat_midpoint = (flat_lower + flat_upper) / 2
        flat_outcome = e021.batch_release(flat_models, flat_data, flat_params,
                                          flat_z1, flat_z2, flat_midpoint)
        jax.block_until_ready(flat_outcome[0])
        for probe_index, (name, trace, final) in enumerate(zip(names, traces, final_states)):
            section = slice(probe_index * count_worlds, (probe_index + 1) * count_worlds)
            _, _, _, path, work, peak = final
            lower, upper = flat_lower[section], flat_upper[section]
            midpoint = flat_midpoint[section]
            values = {
                f"{name}__trace": np.asarray(trace, dtype=np.float32),
                f"{name}__interval_low": np.asarray(lower, dtype=np.float32),
                f"{name}__interval_high": np.asarray(upper, dtype=np.float32),
                f"{name}__oracle_speed": np.asarray(midpoint, dtype=np.float32),
                f"{name}__oracle_error": np.abs(np.asarray(flat_outcome[0][section])
                                                  - np.asarray(goals)),
                f"{name}__target_attained": np.asarray(flat_outcome[4][section]),
                f"{name}__stopped": np.asarray(flat_outcome[5][section]),
                f"{name}__path_displacement": np.asarray(path, dtype=np.float32),
                f"{name}__absolute_work": np.asarray(work, dtype=np.float32),
                f"{name}__peak_speed": np.asarray(peak, dtype=np.float32),
            }
            for key, value in values.items():
                collected.setdefault(key, []).append(value)
        print(json.dumps({"built": start + len(worlds), "count": count,
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    output = {key: np.concatenate(parts) for key, parts in collected.items()}
    output["goal"] = np.asarray([world.goal for world in worlds_all], dtype=np.float32)
    output["parameters"] = np.asarray(
        [[getattr(world, key) for key in PARAMETERS] for world in worlds_all], dtype=np.float32)
    output["world_seed"] = np.asarray([world.seed for world in worlds_all], dtype=np.int64)
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "data")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    smoke = smoke_monotonicity()
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "audit" / "smoke_monotonicity.json").write_text(
        json.dumps(smoke, indent=2) + "\n")
    if not smoke["passed"]:
        raise RuntimeError("excluded smoke grid did not validate monotonicity")
    if args.smoke_only:
        print(json.dumps(smoke, indent=2))
        return
    manifest = {"status": "generated_before_modeling", "environment": {
        "mujoco": mujoco.__version__, "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()]},
        "parameter_order": PARAMETERS, "bisection_steps": BISECTION_STEPS,
        "smoke_seed": SMOKE_SEED, "smoke": smoke,
        "goal_grid_m": GOAL_GRID.tolist(), "splits": {}}
    for split, (seed, count) in SPLITS.items():
        started = time.perf_counter()
        arrays = generate_split(seed, count, args.chunk_size)
        if split == "test":
            target_suffixes = ("__interval_low", "__interval_high", "__oracle_speed",
                               "__oracle_error", "__target_attained", "__stopped")
            target_keys = {key for key in arrays if key.endswith(target_suffixes)}
            target_keys.add("parameters")
            inputs = {key: value for key, value in arrays.items() if key not in target_keys}
            targets = {key: value for key, value in arrays.items()
                       if key in target_keys or key in ("goal", "world_seed")}
            path = args.output / "test_inputs.npz"
            target_path = HERE / "audit" / "test_targets.npz"
            np.savez_compressed(path, **inputs)
            np.savez_compressed(target_path, **targets)
            target_hash = sha256(target_path)
        else:
            path = args.output / f"{split}.npz"
            np.savez_compressed(path, **arrays)
            target_hash = None
        manifest["splits"][split] = {"seed": seed, "rows": count,
            "elapsed_seconds": time.perf_counter() - started, "sha256": sha256(path),
            "evaluator_target_sha256": target_hash}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
