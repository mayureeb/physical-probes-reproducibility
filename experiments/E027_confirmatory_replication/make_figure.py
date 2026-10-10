#!/usr/bin/env python3
"""Plot actual completion and paired effects from the confirmatory replication."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

ROWS = (
    ("Positive vs. multisine, 0.10 m", "positive_minus_multisine_at_0.10m"),
    ("Negative vs. multisine, 0.10 m", "negative_minus_multisine_at_0.10m"),
    ("Positive vs. multisine, 0.20 m", "positive_minus_multisine_at_0.20m"),
    ("Negative vs. multisine, 0.20 m", "negative_minus_multisine_at_0.20m"),
    ("Multisine, 17 vs. 9 observations", "multisine_17_minus_9_samples"),
    ("Positive, 9 vs. 401 observations", "positive_9_minus_401_samples"),
    ("True vs. estimated parameters", "true_parameters_minus_structured"),
)


def main() -> None:
    result = json.loads((HERE / "results.json").read_text())
    records = [result["comparisons"][key] for _, key in ROWS]
    labels = [label for label, _ in ROWS]
    y = np.arange(len(ROWS))

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5,
                         "axes.labelsize": 8.5, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.55), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1.05, 1.0)})

    ax = axes[0]
    first = 100 * np.array([record["first_completion"] for record in records])
    second = 100 * np.array([record["second_completion"] for record in records])
    for index in y:
        ax.plot([second[index], first[index]], [index, index], color="0.72", lw=1.5)
    ax.scatter(first, y, color="#2f6fb0", marker="o", s=28, label="First condition", zorder=3)
    ax.scatter(second, y, color="#cc6b19", marker="s", s=25, label="Reference", zorder=3)
    ax.set_yticks(y, labels, fontsize=7.2)
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.92")
    ax.set(title="(a) Completion", xlabel="Physical completion (%)")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    point = 100 * np.array([record["difference"] for record in records])
    interval = 100 * np.array([record["bootstrap_95_interval"] for record in records])
    ax.errorbar(point, y,
                xerr=np.vstack((point - interval[:, 0], interval[:, 1] - point)),
                fmt="o", color="#333333", ecolor="#777777", capsize=2.5, ms=4, lw=1.1)
    ax.axvline(0, color="0.35", ls="--", lw=0.9)
    for index, value in enumerate(point):
        ax.annotate(f"{value:.2f}", (value, index), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=7)
    ax.set_yticks(y, [""] * len(y))
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.92")
    ax.set(title="(b) Paired differences ($N=4{,}096$)",
           xlabel="Completion difference (percentage points)", xlim=(-1.5, 30.5))

    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "confirmatory_replication.pdf", bbox_inches="tight")
    fig.savefig(output / "confirmatory_replication.png", dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
