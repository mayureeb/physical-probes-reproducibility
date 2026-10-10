#!/usr/bin/env python3
"""Run fresh-seed full-sequence MJX acceptance with separate status fields."""
from __future__ import annotations

import hashlib
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
sys.path.insert(0, str(HERE.parent / "E020_mjx_full_sequence"))
import run as e020  # noqa: E402


def one_release(model, data, material, initial_z1, initial_z2, release_speed):
    gain, strength, tau1, tau2, mix = material

    def accelerate(carry, _):
        current, z1, z2, attained = carry
        active = ~attained
        force, next_z1, next_z2 = e020.applied_force(
            current, strength, tau1, tau2, mix, z1, z2
        )
        ctrl = current.ctrl.at[0].set(jp.clip(
            gain * (release_speed - current.qvel[0]), 0, 50
        ))
        advanced = mjx.step(model, current.replace(qfrc_applied=force, ctrl=ctrl))
        chosen = e020.select(active, advanced, current)
        z1 = jp.where(active, next_z1, z1)
        z2 = jp.where(active, next_z2, z2)
        attained = attained | (chosen.qvel[0] >= release_speed)
        return (chosen, z1, z2, attained), None

    accelerated, _ = jax.lax.scan(
        accelerate, (data, initial_z1, initial_z2, data.qvel[0] >= release_speed),
        None, length=e020.ACCEL_STEPS
    )
    release_data, z1, z2, attained = accelerated
    release_position = release_data.qpos[0]
    release_data = release_data.replace(ctrl=release_data.ctrl.at[0].set(0.0))

    def stop(carry, _):
        current, state1, state2, stopped, used = carry
        active = ~stopped
        force, next_z1, next_z2 = e020.applied_force(
            current, strength, tau1, tau2, mix, state1, state2
        )
        advanced = mjx.step(model, current.replace(qfrc_applied=force))
        chosen = e020.select(active, advanced, current)
        state1 = jp.where(active, next_z1, state1)
        state2 = jp.where(active, next_z2, state2)
        stopped = stopped | (jp.abs(chosen.qvel[0]) < 1e-4)
        return (chosen, state1, state2, stopped, used + active.astype(jp.int32)), None

    final_state, _ = jax.lax.scan(
        stop, (release_data, z1, z2, False, jp.int32(0)),
        None, length=e020.STOP_STEPS
    )
    final, z1, z2, stopped, used = final_state
    return (final.qpos[0] - release_position, final.qvel[0], z1, z2,
            attained, stopped, used)


@jax.jit
def batch_release(models, data, params, z1, z2, release_speed):
    material = (params["gain"], params["strength"], params["tau1"],
                params["tau2"], params["mix"])
    return jax.vmap(one_release, in_axes=(0, 0, 0, 0, 0, 0))(
        models, data, material, z1, z2, release_speed
    )


def batch_oracle(models, data, params, z1, z2, goals):
    low, high = jp.zeros_like(goals), jp.full_like(goals, 3.0)
    for _ in range(14):
        middle = (low + high) / 2
        distance, *_ = batch_release(models, data, params, z1, z2, middle)
        low = jp.where(distance < goals, middle, low)
        high = jp.where(distance < goals, high, middle)
    speed = (low + high) / 2
    return speed, batch_release(models, data, params, z1, z2, speed)


def main() -> None:
    worlds = e020.e014.make_worlds(21021, 16, [4])
    _, models, data = e020.make_models(worlds)
    params = e020.parameters(worlds)
    probe_final, trace = e020.batch_probe(
        models, data, params, jp.asarray(e020.probe_targets())
    )
    probe_data, z1, z2 = probe_final
    started = time.perf_counter()
    differences = []
    fixed_records = []
    for speed in (0.25, 0.75, 1.25):
        requested = jp.full((len(worlds),), speed, dtype=jp.float32)
        outcome = batch_release(models, probe_data, params, z1, z2, requested)
        jax.block_until_ready(outcome[0])
        distance = np.asarray(outcome[0])
        reference = np.asarray([
            e020.e014.simulate(world, speed, peak=e020.PROBE_PEAK, render=False)["distance"]
            for world in worlds
        ])
        difference = np.abs(distance - reference)
        differences.extend(difference.tolist())
        fixed_records.append({
            "speed_mps": speed,
            "absolute_distance_difference_m": difference.tolist(),
            "target_attained": np.asarray(outcome[4]).tolist(),
            "stopped_within_horizon": np.asarray(outcome[5]).tolist(),
            "stopping_steps_used": np.asarray(outcome[6]).tolist(),
        })
    goals = jp.asarray([world.goal for world in worlds], dtype=jp.float32)
    speed, outcome = batch_oracle(models, probe_data, params, z1, z2, goals)
    jax.block_until_ready(outcome[0])
    elapsed = time.perf_counter() - started
    speed_np = np.asarray(speed)
    reference_speed = np.asarray([e020.standard_oracle(world) for world in worlds])
    speed_difference = np.abs(speed_np - reference_speed)
    oracle_error = np.abs(np.asarray(outcome[0]) - np.asarray([world.goal for world in worlds]))
    differences_np = np.asarray(differences)
    finite = bool(np.isfinite(np.concatenate((differences_np, speed_np,
                                              oracle_error, np.asarray(trace).ravel()))).all())
    checks = {
        "finite": finite,
        "fixed_distance_agreement_count_at_0p02m": int(np.sum(differences_np <= 0.02)),
        "fixed_distance_agreement_required": 47,
        "fixed_distance_median_difference_m": float(np.median(differences_np)),
        "fixed_distance_median_required_m": 0.01,
        "oracle_speed_agreement_count_at_0p02mps": int(np.sum(speed_difference <= 0.02)),
        "oracle_speed_agreement_required": 15,
        "oracle_completion_count": int(np.sum(oracle_error <= e020.e014.TOLERANCE)),
        "oracle_completion_required": 15,
        "oracle_stopping_count": int(np.sum(np.asarray(outcome[5]))),
        "oracle_stopping_required": 15,
    }
    checks["passed"] = bool(
        finite and checks["fixed_distance_agreement_count_at_0p02m"] >= 47
        and checks["fixed_distance_median_difference_m"] <= 0.01
        and checks["oracle_speed_agreement_count_at_0p02mps"] >= 15
        and checks["oracle_completion_count"] >= 15
        and checks["oracle_stopping_count"] >= 15
    )
    result = {
        "status": "e021_locked_result",
        "environment": {"mujoco": mujoco.__version__, "jax": jax.__version__,
                        "devices": [str(device) for device in jax.devices()]},
        "compile_and_execution_seconds": elapsed,
        "checks": checks,
        "fixed_speed_records": fixed_records,
        "oracle": {
            "mjx_speed_mps": speed_np.tolist(),
            "mujoco_speed_mps": reference_speed.tolist(),
            "absolute_speed_difference_mps": speed_difference.tolist(),
            "landing_error_m": oracle_error.tolist(),
            "target_attained": np.asarray(outcome[4]).tolist(),
            "stopped_within_horizon": np.asarray(outcome[5]).tolist(),
            "stopping_steps_used": np.asarray(outcome[6]).tolist(),
        },
        "source_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("PROTOCOL.md", "run.py")
        },
    }
    output = HERE / "audit" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
