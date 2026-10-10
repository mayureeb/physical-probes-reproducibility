#!/usr/bin/env python3
"""Plot the planning ladder and all registered paired effects."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROBES = (("half_then_full_forward", "Half then full"),
          ("fixed_negative_first", "Negative first"))
METHODS = (("direct", "Direct trace decoder"),
           ("learned_action", "Estimated state + learned map"),
           ("estimated_planner", "Estimated state + planner"))
COLORS = ("#4c78a8", "#f58518", "#54a24b")
EFFECTS = (
    ("Half then full: planner $-$ learned map",
     "half_then_full_forward__estimated_planner_minus_learned_action"),
    ("Negative first: planner $-$ learned map",
     "fixed_negative_first__estimated_planner_minus_learned_action"),
    ("Half then full: true $-$ estimated state",
     "half_then_full_forward__true_minus_estimated_planner"),
    ("Negative first: true $-$ estimated state",
     "fixed_negative_first__true_minus_estimated_planner"),
)


def main() -> None:
    result = json.loads((HERE / "results.json").read_text())
    completion = np.array([[100 * result["descriptive"][f"{p}__{m}"]["completion"]
                            for m, _ in METHODS] for p, _ in PROBES])
    all_methods = METHODS + (("true_planner", "True state + planner"),)
    error = np.array([[result["descriptive"][f"{p}__{m}"]["mean_landing_error_m"]
                       for m, _ in all_methods] for p, _ in PROBES])

    plt.rcParams.update({"font.size": 8.2, "axes.titlesize": 9.2,
                         "axes.labelsize": 8.2, "legend.fontsize": 7.2})
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": (0.9, 0.9, 1.35)})
    x = np.arange(2)
    offsets = np.linspace(-0.18, 0.18, len(METHODS))

    ax = axes[0]
    for i, ((_, label), color, offset) in enumerate(zip(METHODS, COLORS, offsets)):
        ax.scatter(x + offset, completion[:, i], color=color, s=27, label=label, zorder=3)
    for probe_index in range(2):
        ax.plot(x[probe_index] + offsets, completion[probe_index], color="0.80", lw=1)
    ax.text(0.5, 0.96, "True-state planner = 100% (both)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.2,
            bbox={"facecolor": "#f6eef5", "edgecolor": "#b279a2", "pad": 2})
    ax.set(title="(a) Practical completion", ylabel="Completion at 0.06 m (%)",
           xticks=x, xticklabels=("Half/full", "Negative"), ylim=(18, 34))

    ax = axes[1]
    all_colors = COLORS + ("#b279a2",)
    all_offsets = np.linspace(-0.24, 0.24, len(all_methods))
    for i, ((_, label), color, offset) in enumerate(zip(all_methods, all_colors, all_offsets)):
        ax.scatter(x + offset, error[:, i], color=color, s=27, label=label, zorder=3)
    for probe_index in range(2):
        ax.plot(x[probe_index] + all_offsets, error[probe_index], color="0.80", lw=1)
    ax.set_yscale("log")
    ax.set(title="(b) Landing error", ylabel="Mean absolute error (m, log scale)",
           xticks=x, xticklabels=("Half/full", "Negative"))

    ax = axes[2]
    records = [result["comparisons"][key] for _, key in EFFECTS]
    point = 100 * np.array([r["difference"] for r in records])
    interval = 100 * np.array([r["bootstrap_95_interval"] for r in records])
    y = np.arange(len(records))
    ax.errorbar(point, y, xerr=np.vstack((point - interval[:, 0], interval[:, 1] - point)),
                fmt="o", color="#333333", ecolor="#777777", capsize=2.5, ms=4, lw=1.1)
    ax.axvline(0, color="0.35", ls="--", lw=0.8)
    ax.set_yticks(y, [label for label, _ in EFFECTS], fontsize=6.8)
    ax.invert_yaxis()
    ax.set(title="(c) Registered paired effects",
           xlabel="Completion difference (pp)")
    for i, value in enumerate(point):
        ax.annotate(f"{value:.2f}", (value, i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=6.8)

    for ax in axes:
        ax.grid(axis="y", color="0.93")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "model_based_planning.pdf", bbox_inches="tight")
    fig.savefig(output / "model_based_planning.png", dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
