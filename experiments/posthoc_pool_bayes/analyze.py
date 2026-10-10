#!/usr/bin/env python3
"""Compute same-goal finite-pool posterior actions under the matched noise model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E030_decision_directed_probe"
sys.path.insert(0, str(SOURCE))
import fit_and_freeze as e030  # noqa: E402

PROBES = ("decision_directed", "d_optimal", "fixed_positive_first",
          "fixed_negative_first")
BOOTSTRAPS = 20_000


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def optimal_action(low: np.ndarray, high: np.ndarray,
                   weights: np.ndarray) -> tuple[float, float]:
    endpoints = np.unique(np.concatenate((low, high)))
    grid = np.unique(np.concatenate((endpoints, (endpoints[:-1] + endpoints[1:]) / 2)))
    covered = (grid[:, None] >= low[None]) & (grid[:, None] <= high[None])
    mass = covered @ weights
    best_mass = mass.max()
    tied = grid[np.isclose(mass, best_mass, rtol=0, atol=1e-12)]
    posterior_midpoint = float(np.sum(weights * (low + high) / 2))
    action = float(tied[np.argmin(np.abs(tied - posterior_midpoint))])
    return action, float(best_mass)


def paired(first: np.ndarray, second: np.ndarray, seed: int) -> dict:
    delta = first.astype(np.int8) - second.astype(np.int8)
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAPS, dtype=np.float32)
    for start in range(0, BOOTSTRAPS, 500):
        index = rng.integers(0, len(delta), size=(min(500, BOOTSTRAPS-start), len(delta)))
        samples[start:start + len(index)] = delta[index].mean(1)
    discordant = int(np.sum(delta != 0))
    first_only = int(np.sum(delta == 1))
    p_value = 1.0 if not discordant else float(binomtest(first_only, discordant).pvalue)
    return {"pool_bayes_completion": float(first.mean()),
            "extra_trees_completion": float(second.mean()),
            "difference": float(delta.mean()),
            "bootstrap_95_interval": np.quantile(samples, (0.025, 0.975)).tolist(),
            "discordant_worlds": discordant, "pool_bayes_only": first_only,
            "exact_two_sided_p": p_value}


def main() -> None:
    train = load(SOURCE / "data/train.npz")
    validation = load(SOURCE / "data/validation.npz")
    test = load(SOURCE / "data/test_inputs.npz")
    targets = load(SOURCE / "audit/test_targets.npz")
    frozen = load(SOURCE / "audit/estimates_and_direct_actions.npz")
    reference = {key: np.concatenate((train[key], validation[key]))
                 for key in train if key in validation}
    covariance = 0.04 ** 2 * 0.8 ** np.abs(
        np.arange(len(e030.TRACE_INDEX))[:, None] - np.arange(len(e030.TRACE_INDEX))[None])
    force_inverse = np.linalg.inv(covariance)
    results, output = {}, {}
    for probe_index, probe in enumerate(PROBES):
        reference_trace = reference[f"{probe}__trace"][:, e030.TRACE_INDEX]
        observed = e030.observed(test, probe, "test")
        actions = np.empty(len(observed), dtype=np.float32)
        masses = np.empty(len(observed), dtype=np.float32)
        effective = np.empty(len(observed), dtype=np.float32)
        maximum_weight = np.empty(len(observed), dtype=np.float32)
        for index in range(len(observed)):
            candidates = np.flatnonzero(reference["goal"] == test["goal"][index])
            delta = reference_trace[candidates] - observed[index]
            squared = np.sum((delta[:, :, 0] / 0.01) ** 2, axis=1)
            squared += np.einsum("ni,ij,nj->n", delta[:, :, 1], force_inverse,
                                 delta[:, :, 1])
            log_weight = -0.5 * squared
            log_weight -= log_weight.max()
            weight = np.exp(log_weight)
            weight /= weight.sum()
            low = reference[f"{probe}__interval_low"][candidates]
            high = reference[f"{probe}__interval_high"][candidates]
            actions[index], masses[index] = optimal_action(low, high, weight)
            effective[index] = 1.0 / np.sum(weight ** 2)
            maximum_weight[index] = weight.max()
        low = targets[f"{probe}__interval_low"]
        high = targets[f"{probe}__interval_high"]
        pool_success = (actions >= low) & (actions <= high)
        tree_action = frozen[f"{probe}__direct"]
        tree_success = (tree_action >= low) & (tree_action <= high)
        results[probe] = {"comparison": paired(pool_success, tree_success, 313200 + probe_index),
            "mean_claimed_posterior_success_mass": float(masses.mean()),
            "median_effective_pool_size": float(np.median(effective)),
            "median_maximum_posterior_weight": float(np.median(maximum_weight))}
        output[f"{probe}__pool_bayes_action"] = actions
        output[f"{probe}__posterior_success_mass"] = masses
        print(json.dumps({"probe": probe, **results[probe]}), flush=True)
    np.savez_compressed(HERE / "predictions.npz", **output)
    result = {"status": "posthoc_finite_pool_bayes", "reference_worlds": len(reference["goal"]),
              "same_goal_candidates": int(np.sum(reference["goal"] == reference["goal"][0])),
              "test_worlds": len(test["goal"]), "probes": results}
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
