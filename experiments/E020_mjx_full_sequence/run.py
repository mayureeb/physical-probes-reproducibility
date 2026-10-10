#!/usr/bin/env python3
"""Validate the complete E014 sequence with batched MJX/JAX dynamics."""
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
sys.path.insert(0, str(HERE.parent / "E014_mujoco_learned_transfer"))
import simulator as e014  # noqa: E402

PROBE_PEAK = 1.2
ACCEL_STEPS = 2500
STOP_STEPS = 15000


def probe_targets(peak: float = PROBE_PEAK) -> np.ndarray:
    leg = np.arange(0, peak + 1e-9, e014.ACCEL * e014.DT)
    return np.concatenate((leg, leg[-2::-1], -leg[1:], -leg[-2::-1], np.zeros(100))).astype(np.float32)


def make_models(worlds):
    standard = [e014.model(world) for world in worlds]
    converted = [mjx.put_model(model) for model in standard]
    data = [mjx.make_data(model) for model in converted]
    stack = lambda *leaves: jp.stack(leaves)
    return standard, jax.tree.map(stack, *converted), jax.tree.map(stack, *data)


def parameters(worlds):
    names = ("gain", "strength", "tau1", "tau2", "mix")
    return {name: jp.asarray([getattr(world, name) for world in worlds], dtype=jp.float32)
            for name in names}


def select(flag, new, old):
    return jax.tree.map(lambda x, y: jp.where(flag, x, y), new, old)


def update_state(z, target, tau):
    return target + (z - target) * jp.exp(-e014.DT / tau)


def applied_force(data, strength, tau1, tau2, mix, z1, z2):
    speed = jp.abs(data.qvel[0])
    next_z1 = update_state(z1, speed, tau1)
    next_z2 = update_state(z2, speed, tau2)
    extra = strength * ((1 - mix) * next_z1 + mix * next_z2)
    force = data.qfrc_applied.at[0].set(-jp.sign(data.qvel[0]) * extra)
    return force, next_z1, next_z2


def one_probe(model, data, material, targets):
    gain, strength, tau1, tau2, mix = material

    def step(carry, target):
        current, z1, z2 = carry
        force, next_z1, next_z2 = applied_force(
            current, strength, tau1, tau2, mix, z1, z2
        )
        ctrl = current.ctrl.at[0].set(jp.clip(gain * (target - current.qvel[0]), -50, 50))
        advanced = mjx.step(model, current.replace(qfrc_applied=force, ctrl=ctrl))
        observed = jp.stack((advanced.qvel[0], advanced.ctrl[0] + advanced.qfrc_constraint[0]))
        return (advanced, next_z1, next_z2), observed

    return jax.lax.scan(step, (data, jp.float32(0), jp.float32(0)), targets)


def one_release(model, data, material, initial_z1, initial_z2, release_speed):
    gain, strength, tau1, tau2, mix = material

    def accelerate(carry, _):
        current, z1, z2, released = carry
        active = ~released
        force, next_z1, next_z2 = applied_force(
            current, strength, tau1, tau2, mix, z1, z2
        )
        control = jp.clip(gain * (release_speed - current.qvel[0]), 0, 50)
        ctrl = current.ctrl.at[0].set(control)
        advanced = mjx.step(model, current.replace(qfrc_applied=force, ctrl=ctrl))
        chosen = select(active, advanced, current)
        z1 = jp.where(active, next_z1, z1)
        z2 = jp.where(active, next_z2, z2)
        released = released | (chosen.qvel[0] >= release_speed)
        return (chosen, z1, z2, released), None

    accelerated, _ = jax.lax.scan(
        accelerate, (data, initial_z1, initial_z2, data.qvel[0] >= release_speed),
        None, length=ACCEL_STEPS
    )
    release_data, z1, z2, released = accelerated
    release_position = release_data.qpos[0]
    release_data = release_data.replace(ctrl=release_data.ctrl.at[0].set(0.0))

    def stop(carry, _):
        current, state1, state2, stopped, used = carry
        active = ~stopped
        force, next_z1, next_z2 = applied_force(
            current, strength, tau1, tau2, mix, state1, state2
        )
        advanced = mjx.step(model, current.replace(qfrc_applied=force))
        chosen = select(active, advanced, current)
        state1 = jp.where(active, next_z1, state1)
        state2 = jp.where(active, next_z2, state2)
        stopped = stopped | (jp.abs(chosen.qvel[0]) < 1e-4)
        used = used + active.astype(jp.int32)
        return (chosen, state1, state2, stopped, used), None

    stopped_state, _ = jax.lax.scan(
        stop, (release_data, z1, z2, False, jp.int32(0)), None, length=STOP_STEPS
    )
    final, z1, z2, stopped, used = stopped_state
    return final.qpos[0] - release_position, final.qvel[0], z1, z2, stopped & released, used


@jax.jit
def batch_probe(models, data, params, targets):
    material = (params["gain"], params["strength"], params["tau1"],
                params["tau2"], params["mix"])
    return jax.vmap(one_probe, in_axes=(0, 0, 0, None))(models, data, material, targets)


@jax.jit
def batch_release(models, data, params, z1, z2, release_speed):
    material = (params["gain"], params["strength"], params["tau1"],
                params["tau2"], params["mix"])
    return jax.vmap(one_release, in_axes=(0, 0, 0, 0, 0, 0))(
        models, data, material, z1, z2, release_speed
    )


def batch_oracle(models, probe_data, params, z1, z2, goals):
    low, high = jp.zeros_like(goals), jp.full_like(goals, 3.0)
    for _ in range(14):
        middle = (low + high) / 2
        distance, *_ = batch_release(models, probe_data, params, z1, z2, middle)
        low = jp.where(distance < goals, middle, low)
        high = jp.where(distance < goals, high, middle)
    speed = (low + high) / 2
    outcome = batch_release(models, probe_data, params, z1, z2, speed)
    return speed, outcome


def standard_oracle(world) -> float:
    low, high = 0.0, 3.0
    for _ in range(14):
        middle = (low + high) / 2
        if e014.simulate(world, middle, peak=PROBE_PEAK, render=False)["distance"] < world.goal:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def main() -> None:
    worlds = e014.make_worlds(20020, 8, [4])
    _, models, data = make_models(worlds)
    params = parameters(worlds)
    targets = jp.asarray(probe_targets())
    started = time.perf_counter()
    probe_final, trace = batch_probe(models, data, params, targets)
    probe_data, z1, z2 = probe_final
    fixed_records = []
    fixed_differences = []
    all_stopped = []
    for speed in (0.25, 0.75, 1.25):
        speeds = jp.full((len(worlds),), speed, dtype=jp.float32)
        outcome = batch_release(models, probe_data, params, z1, z2, speeds)
        jax.block_until_ready(outcome[0])
        mjx_distance = np.asarray(outcome[0])
        reference = np.asarray([
            e014.simulate(world, speed, peak=PROBE_PEAK, render=False)["distance"]
            for world in worlds
        ])
        difference = np.abs(mjx_distance - reference)
        fixed_differences.extend(difference.tolist())
        all_stopped.extend(np.asarray(outcome[4]).tolist())
        fixed_records.append({"speed": speed, "mjx_distance_m": mjx_distance.tolist(),
                              "mujoco_distance_m": reference.tolist(),
                              "absolute_difference_m": difference.tolist()})
    goals = jp.asarray([world.goal for world in worlds], dtype=jp.float32)
    mjx_speed, mjx_outcome = batch_oracle(models, probe_data, params, z1, z2, goals)
    jax.block_until_ready(mjx_outcome[0])
    elapsed = time.perf_counter() - started
    mjx_speed_np = np.asarray(mjx_speed)
    reference_speed = np.asarray([standard_oracle(world) for world in worlds])
    speed_difference = np.abs(mjx_speed_np - reference_speed)
    oracle_distance = np.asarray(mjx_outcome[0])
    oracle_error = np.abs(oracle_distance - np.asarray([world.goal for world in worlds]))
    all_stopped.extend(np.asarray(mjx_outcome[4]).tolist())
    finite_values = np.concatenate((np.asarray(fixed_differences), mjx_speed_np,
                                    oracle_distance, np.asarray(trace).ravel()))
    fixed_differences_np = np.asarray(fixed_differences)
    checks = {
        "finite": bool(np.isfinite(finite_values).all()),
        "fixed_distance_agreement_count_at_0p02m": int(np.sum(fixed_differences_np <= 0.02)),
        "fixed_distance_agreement_required": 23,
        "fixed_distance_median_difference_m": float(np.median(fixed_differences_np)),
        "fixed_distance_median_required_m": 0.01,
        "oracle_speed_agreement_count_at_0p02mps": int(np.sum(speed_difference <= 0.02)),
        "oracle_speed_agreement_required": 7,
        "mjx_oracle_completion_count": int(np.sum(oracle_error <= e014.TOLERANCE)),
        "mjx_oracle_completion_required": 7,
        "all_releases_stopped": bool(np.all(all_stopped)),
    }
    checks["passed"] = bool(
        checks["finite"]
        and checks["fixed_distance_agreement_count_at_0p02m"] >= 23
        and checks["fixed_distance_median_difference_m"] <= 0.01
        and checks["oracle_speed_agreement_count_at_0p02mps"] >= 7
        and checks["mjx_oracle_completion_count"] >= 7
        and checks["all_releases_stopped"]
    )
    result = {
        "status": "e020_locked_result",
        "environment": {"mujoco": mujoco.__version__, "jax": jax.__version__,
                        "devices": [str(device) for device in jax.devices()]},
        "compile_and_execution_seconds": elapsed,
        "checks": checks,
        "fixed_speed_records": fixed_records,
        "oracle": {"mjx_speed_mps": mjx_speed_np.tolist(),
                   "mujoco_speed_mps": reference_speed.tolist(),
                   "absolute_speed_difference_mps": speed_difference.tolist(),
                   "mjx_distance_m": oracle_distance.tolist(),
                   "landing_error_m": oracle_error.tolist()},
        "trace_shape": list(np.asarray(trace).shape),
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
