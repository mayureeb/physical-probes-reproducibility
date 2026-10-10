#!/usr/bin/env python3
"""Plot temporal-budget outcomes and registered paired effects."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BUDGETS = np.array([9, 33, 101, 401])
EFFECTS = (("33 minus 9 observations", "budget_33_minus_9"),
           ("401 minus 9 observations", "budget_401_minus_9"),
           ("True state minus 401 observations", "true_state_minus_401"))


def wilson_half(p: np.ndarray, n: int) -> np.ndarray:
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return np.vstack((p - (center - half), center + half - p))


def main() -> None:
    result = json.loads((HERE / "results.json").read_text())
    completion = np.array([result["descriptive"][f"budget_{b}"]["completion"]
                           for b in BUDGETS])
    landing = np.array([result["descriptive"][f"budget_{b}"]["mean_landing_error_m"]
                        for b in BUDGETS])
    parameter = np.array([result["parameter_nmse"][str(b)] for b in BUDGETS])

    plt.rcParams.update({"font.size": 8.2, "axes.titlesize": 9.2,
                         "axes.labelsize": 8.2, "legend.fontsize": 7.2})
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.05), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1, 1, 1.25)})
    x = np.arange(len(BUDGETS))

    ax = axes[0]
    ax.errorbar(x, 100 * completion, yerr=100 * wilson_half(completion, result["worlds"]),
                color="#4c78a8", marker="o", capsize=2.5, lw=1.4)
    for i, value in enumerate(100 * completion):
        ax.annotate(f"{value:.2f}", (i, value), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7)
    ax.text(0.5, 0.05, "True-state planner = 100%", transform=ax.transAxes,
            ha="center", fontsize=7.2,
            bbox={"facecolor": "#f6eef5", "edgecolor": "#b279a2", "pad": 2})
    ax.set(title="(a) Physical completion", xlabel="Temporal observations",
           ylabel="Completion at 0.06 m (%)", xticks=x, xticklabels=BUDGETS,
           ylim=(27.5, 36.5))

    ax = axes[1]
    ax.plot(x, 100 * parameter / parameter[0], marker="o", color="#f58518",
            label="Parameter NMSE")
    ax.plot(x, 100 * landing / landing[0], marker="s", color="#54a24b",
            label="Landing error")
    ax.axhline(100, color="0.55", ls="--", lw=0.8)
    ax.set(title="(b) Error relative to 9 samples", xlabel="Temporal observations",
           ylabel="Relative error (%)", xticks=x, xticklabels=BUDGETS)
    ax.legend(frameon=False)

    ax = axes[2]
    records = [result["comparisons"][key] for _, key in EFFECTS]
    point = 100 * np.array([record["difference"] for record in records])
    interval = 100 * np.array([record["bootstrap_95_interval"] for record in records])
    y = np.arange(len(records))
    ax.errorbar(point, y, xerr=np.vstack((point - interval[:, 0], interval[:, 1] - point)),
                fmt="o", color="#333333", ecolor="#777777", capsize=2.5, ms=4, lw=1.1)
    ax.axvline(0, color="0.35", ls="--", lw=0.8)
    ax.set_yticks(y, [label for label, _ in EFFECTS], fontsize=6.8)
    ax.invert_yaxis()
    ax.set(title="(c) Registered paired effects", xlabel="Completion difference (pp)")
    for i, value in enumerate(point):
        ax.annotate(f"{value:.2f}", (value, i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7)

    for ax in axes:
        ax.grid(axis="y", color="0.93")
        ax.spines[["top", "right"]].set_visible(False)
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "temporal_state_estimation.pdf", bbox_inches="tight")
    fig.savefig(output / "temporal_state_estimation.png", dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
