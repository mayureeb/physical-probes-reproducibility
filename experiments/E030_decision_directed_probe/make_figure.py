#!/usr/bin/env python3
"""Plot probe selection and untouched-test results from frozen aggregates."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROBES = (
    ("decision_directed", "Task-selected", "#009E73", "o"),
    ("d_optimal", "D-optimal multisine", "#CC79A7", "^"),
    ("fixed_positive_first", "Positive-first", "#D55E00", "s"),
    ("fixed_negative_first", "Negative-first", "#0072B2", "D"),
)


def wilson_half(values: np.ndarray, count: int) -> np.ndarray:
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (values + z * z / (2 * count)) / denominator
    half = z * np.sqrt(values * (1 - values) / count + z * z / (4 * count * count)) / denominator
    return np.vstack((values - (center - half), center + half - values))


def main() -> None:
    selection = json.loads((HERE / "audit/selection_lock.json").read_text())
    result = json.loads((HERE / "results.json").read_text())
    with np.load(HERE / "audit/selected_waveforms.npz", allow_pickle=False) as data:
        waveforms = {key: data[key] for key in data.files}

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True)

    ax = axes[0]
    time = np.arange(len(waveforms["decision_directed"])) * 0.002
    for key, label, color, _ in PROBES[:2]:
        ax.plot(time, waveforms[key], color=color, linewidth=1.5, label=label)
    ax.axvspan(0.602, 0.802, color="0.94", zorder=-1)
    ax.text(0.702, -1.08, "rest", ha="center", va="bottom", color="0.45", fontsize=7)
    ax.set(xlabel="Probe time (s)", ylabel="Commanded speed (m/s)",
           ylim=(-1.25, 1.25))
    ax.set_title("(a) Selected probes", loc="left", fontweight="bold")
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    eligible = selection["eligible_candidates"]
    information = np.array([selection["records"][name]["d_optimal_score"]
                            for name in eligible])
    completion = 100 * np.array([
        selection["records"][name]["decision_score"]["completion"]
        for name in eligible
    ])
    ax.scatter(information, completion, s=14, color="0.62", alpha=0.75,
               edgecolors="none")
    sources = {
        "decision_directed": selection["decision_directed_source"],
        "d_optimal": selection["d_optimal_source"],
        "fixed_positive_first": "fixed_positive_first",
        "fixed_negative_first": "fixed_negative_first",
    }
    for key, label, color, marker in PROBES:
        record = selection["records"][sources[key]]
        point_x = record["d_optimal_score"]
        point_y = 100 * record["decision_score"]["completion"]
        ax.scatter(point_x, point_y,
                   s=38, color=color, marker=marker, edgecolors="white",
                   linewidths=0.5, zorder=3)
        offsets = {
            "decision_directed": (5, 4, "left"),
            "d_optimal": (-5, 4, "right"),
            "fixed_positive_first": (5, -9, "left"),
            "fixed_negative_first": (5, 4, "left"),
        }
        dx, dy, align = offsets[key]
        ax.annotate(label, (point_x, point_y), xytext=(dx, dy),
                    textcoords="offset points", ha=align, fontsize=6.2,
                    color="0.12")
    correlation = spearmanr(information, completion).statistic
    ax.text(0.97, 0.95, rf"Spearman $\rho={correlation:.2f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    ax.set(xlabel="Parameter-information score",
           ylabel="Design completion (%)")
    ax.set_title("(b) Objectives disagree", loc="left", fontweight="bold")
    ax.grid(color="0.92", linewidth=0.6)

    ax = axes[2]
    direct = np.array([result["descriptive"][f"{key}__direct"]["completion"]
                       for key, *_ in PROBES])
    planned = np.array([
        result["descriptive"][f"{key}__estimated_planner"]["completion"]
        for key, *_ in PROBES
    ])
    x = np.arange(len(PROBES))
    offsets = (-0.13, 0.13)
    for index, (_, _, color, _) in enumerate(PROBES):
        ax.errorbar(x[index] + offsets[0], 100 * direct[index],
                    yerr=100 * wilson_half(direct[index:index + 1], result["worlds"]),
                    fmt="o", mfc="white", mec=color, mew=1.1, color=color,
                    ecolor=color, capsize=2, ms=4.8,
                    label=None)
        ax.errorbar(x[index] + offsets[1], 100 * planned[index],
                    yerr=100 * wilson_half(planned[index:index + 1], result["worlds"]),
                    fmt="s", mfc=color, mec=color, color=color,
                    ecolor=color, capsize=2, ms=4.4,
                    label=None)
        ax.plot([x[index] + offsets[0], x[index] + offsets[1]],
                [100 * direct[index], 100 * planned[index]],
                color=color, alpha=0.45, linewidth=0.8)
    ax.set_xticks(x, ["Task", "D-optimal\nmultisine", "Positive", "Negative"])
    upper = max(38, 100 * max(direct.max(), planned.max()) + 5)
    ax.set(xlabel="Probe condition", ylabel="Test completion (%)", ylim=(0, upper))
    ax.set_title("(c) Fresh-world test", loc="left", fontweight="bold")
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.text(0.98, 0.95, "True-state planner: 100% for every probe",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="0.15", label="Direct decoder"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="0.15",
               markeredgecolor="0.15", label="Estimated-state planner"),
    ], frameon=False, fontsize=6.5, loc="lower left")

    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "decision_directed_probe.pdf", bbox_inches="tight")
    fig.savefig(output / "decision_directed_probe.png", dpi=300, bbox_inches="tight")
    print(output / "decision_directed_probe.pdf")


if __name__ == "__main__":
    main()
