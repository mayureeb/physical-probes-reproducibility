#!/usr/bin/env python3
"""Development-only differentiable simulator fit on previously opened data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E025 = ROOT / "experiments/E025_decision_ambiguity"
E029 = ROOT / "experiments/E029_temporal_state_estimation"
E028 = ROOT / "experiments/E028_model_based_planning"
sys.path.insert(0, str(E025))
sys.path.insert(0, str(E029))
sys.path.insert(0, str(E028))
import build_data as build  # noqa: E402
import fit_estimators as temporal  # noqa: E402
import plan_before_targets as planning  # noqa: E402

LOW = jnp.asarray((0.6, 0.06, 0.01, 14.0, 0.2, 0.05, 0.7, 0.25))
HIGH = jnp.asarray((1.5, 0.22, 0.12, 30.0, 1.0, 0.25, 1.8, 0.7))
TRACE_INDEX = np.arange(0, 401, 50)
PARAMETERS = ("mass", "mu", "damping", "gain", "strength", "tau1", "tau2", "mix")
SELECTED_FORCE_RHO = 0.8


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def template_model():
    world = build.e020.e014.World(
        family=4, mass=1.0, mu=0.14, damping=0.065, goal=0.6,
        gain=22.0, strength=0.6, tau1=0.15, tau2=1.25, mix=0.475, seed=0)
    _, models, data = build.e020.make_models([world])
    return jax.tree.map(lambda value: value[0], models), jax.tree.map(lambda value: value[0], data)


BASE_MODEL, BASE_DATA = template_model()
BASE_MASS = BASE_MODEL.body_mass[1]
BASE_INERTIA = BASE_MODEL.body_inertia[1]


def model_from_theta(theta):
    mass, mu, damping = theta[:3]
    body_mass = BASE_MODEL.body_mass.at[1].set(mass)
    body_inertia = BASE_MODEL.body_inertia.at[1].set(BASE_INERTIA * mass / BASE_MASS)
    body_subtreemass = BASE_MODEL.body_subtreemass.at[0].set(mass).at[1].set(mass)
    inverse_mass = 1.0 / mass
    body_invweight0 = BASE_MODEL.body_invweight0.at[1, 0].set(inverse_mass)
    dof_invweight0 = BASE_MODEL.dof_invweight0.at[0].set(inverse_mass)
    dof_m0 = BASE_MODEL.dof_M0.at[0].set(mass)
    actuator_acc0 = BASE_MODEL.actuator_acc0.at[0].set(inverse_mass)
    geom_friction = BASE_MODEL.geom_friction.at[:, 0].set(mu)
    dof_damping = BASE_MODEL.dof_damping.at[0].set(damping)
    return BASE_MODEL.replace(body_mass=body_mass, body_inertia=body_inertia,
                              body_subtreemass=body_subtreemass,
                              body_invweight0=body_invweight0,
                              dof_invweight0=dof_invweight0, dof_M0=dof_m0,
                              actuator_acc0=actuator_acc0,
                              geom_friction=geom_friction, dof_damping=dof_damping)


def predict_one(theta, targets):
    model = model_from_theta(theta)
    material = (theta[3], theta[4], theta[5], theta[6], theta[7])
    _, trace = build.one_probe_with_cost(model, BASE_DATA, material, targets)
    return trace[TRACE_INDEX]


predict_batch = jax.jit(jax.vmap(predict_one, in_axes=(0, None)))


def decode(u):
    return LOW + jax.nn.sigmoid(u) * (HIGH - LOW)


def candidate_objective(u, observed, targets):
    theta = decode(u)
    prediction = predict_one(theta, targets)
    residual_velocity = (prediction[:, 0] - observed[:, 0]) / temporal.VELOCITY_SIGMA
    force = prediction[:, 1] - observed[:, 1]
    first = force[:1] / temporal.FORCE_SIGMA
    rest = (force[1:] - SELECTED_FORCE_RHO * force[:-1]) / (
        temporal.FORCE_SIGMA * np.sqrt(1.0 - SELECTED_FORCE_RHO ** 2))
    residual_force = jnp.concatenate((first, rest), axis=0)
    return jnp.mean(residual_velocity ** 2 + residual_force ** 2)


candidate_gradient = jax.jacfwd(candidate_objective, argnums=0)


@jax.jit
def values_and_gradients(u, observed, targets):
    values = jax.vmap(candidate_objective, in_axes=(0, 0, None))(u, observed, targets)
    gradients = jax.vmap(candidate_gradient, in_axes=(0, 0, None))(u, observed, targets)
    return values, gradients


def to_unit(theta: np.ndarray) -> np.ndarray:
    return np.clip((theta - np.asarray(LOW)) / np.asarray(HIGH - LOW), 1e-4, 1.0 - 1e-4)


def fit(observed: np.ndarray, initial_theta: np.ndarray, restarts: int, steps: int, seed: int):
    rng = np.random.default_rng(seed)
    center = np.log(to_unit(initial_theta) / (1.0 - to_unit(initial_theta)))
    offsets = rng.normal(0.0, 0.55, size=(len(observed), restarts, 8)).astype(np.float32)
    offsets[:, 0] = 0.0
    initial = 1.0 / (1.0 + np.exp(-(center[:, None] + offsets)))
    u = jnp.asarray(np.log(initial / (1.0 - initial)).reshape(-1, 8))
    obs = jnp.asarray(np.repeat(observed, restarts, axis=0))
    targets = jnp.asarray(build.probes.library()["fixed_negative_first"])
    first_moment = jnp.zeros_like(u)
    second_moment = jnp.zeros_like(u)
    history = []
    for step in range(1, steps + 1):
        values, gradient = values_and_gradients(u, obs, targets)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient ** 2
        corrected_first = first_moment / (1.0 - 0.9 ** step)
        corrected_second = second_moment / (1.0 - 0.999 ** step)
        rate = 0.03 * 0.5 * (1.0 + np.cos(np.pi * step / steps))
        u = u - rate * corrected_first / (jnp.sqrt(corrected_second) + 1e-8)
        if step == 1 or step % 25 == 0 or step == steps:
            scalar = float(jnp.sum(values))
            history.append((step, scalar))
            print(json.dumps({"step": step, "summed_loss": scalar}), flush=True)
    theta = np.asarray(decode(u)).reshape(len(observed), restarts, 8)
    prediction = np.asarray(predict_batch(decode(u), targets)).reshape(
        len(observed), restarts, len(TRACE_INDEX), 2)
    repeated = observed[:, None]
    velocity = ((prediction[:, :, :, 0] - repeated[:, :, :, 0]) /
                temporal.VELOCITY_SIGMA) ** 2
    force = prediction[:, :, :, 1] - repeated[:, :, :, 1]
    force_white = np.concatenate((force[:, :, :1] / temporal.FORCE_SIGMA,
        (force[:, :, 1:] - SELECTED_FORCE_RHO * force[:, :, :-1]) /
        (temporal.FORCE_SIGMA * np.sqrt(1.0 - SELECTED_FORCE_RHO ** 2))), axis=2) ** 2
    losses = np.mean(velocity + force_white, axis=2)
    selected = np.argmin(losses, axis=1)
    return theta[np.arange(len(theta)), selected], losses[np.arange(len(theta)), selected], history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    test = load(E029 / "data/test_inputs.npz")
    target_file = load(E029 / "audit/test_targets.npz")
    estimates = load(E029 / "audit/estimates_before_targets.npz")
    observed = temporal.observed(test, "test")[:, TRACE_INDEX]
    targets = jnp.asarray(build.probes.library()["fixed_negative_first"])
    true_theta = target_file["parameters"][:args.worlds]
    true_prediction = np.asarray(predict_batch(jnp.asarray(true_theta), targets))
    clean_reference = test["trace"][:args.worlds, TRACE_INDEX]
    initial_theta = estimates["budget_9__estimated_parameters"][:args.worlds]
    estimate, loss, history = fit(observed[:args.worlds], initial_theta,
                                  args.restarts, args.steps, 31031)
    target = true_theta
    scale = target_file["parameters"].std(0) + 1e-8
    goal = test["goal"][:args.worlds]
    seeds = test["world_seed"][:args.worlds]
    initial_action = planning.planned_actions("fixed_negative_first", initial_theta, goal, seeds)
    refined_action = planning.planned_actions("fixed_negative_first", estimate, goal, seeds)
    true_worlds = planning.worlds(true_theta, goal, seeds)
    _, true_models, true_data = build.e020.make_models(true_worlds)
    true_params = build.e020.parameters(true_worlds)
    probe_final, _ = build.batch_probe_with_cost(
        true_models, true_data, true_params,
        jnp.asarray(build.probes.library()["fixed_negative_first"]))
    final_data, z1, z2, *_ = probe_final

    def replay(action):
        outcome = build.e021.batch_release(true_models, final_data, true_params,
                                          z1, z2, jnp.asarray(action))
        distance = np.asarray(outcome[0])
        error = np.abs(distance - goal)
        return {"completion": float(np.mean(error <= 0.06)),
                "mean_landing_error_m": float(np.mean(error))}
    result = {
        "development_only": True,
        "devices": [str(device) for device in jax.devices()],
        "worlds": args.worlds,
        "restarts": args.restarts,
        "steps": args.steps,
        "true_parameter_median_whitened_loss": float(np.median(np.asarray(
            jax.vmap(candidate_objective, in_axes=(0, 0, None))(
                jnp.log((jnp.asarray(true_theta) - LOW) / (HIGH - jnp.asarray(true_theta))),
                jnp.asarray(observed[:args.worlds]), targets)))),
        "true_parameter_max_clean_trace_difference": float(
            np.max(np.abs(true_prediction - clean_reference))),
        "parameter_nmse": float(np.mean(((estimate - target) / scale) ** 2)),
        "initial_parameter_nmse": float(np.mean(((initial_theta - target) / scale) ** 2)),
        "initial_planner": replay(initial_action),
        "refined_planner": replay(refined_action),
        "per_parameter_rmse": {name: float(np.sqrt(np.mean((estimate[:, i] - target[:, i]) ** 2)))
                               for i, name in enumerate(PARAMETERS)},
        "median_whitened_trace_loss": float(np.median(loss)),
        "history": history,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
