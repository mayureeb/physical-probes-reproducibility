#!/usr/bin/env python3
"""Create the consolidated post hoc diagnostic figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROBES = ("passive", "fixed_positive_first", "half_then_full_forward",
          "fixed_negative_first", "multisine", "random_piecewise")
LABELS = {
    "passive": "No probe",
    "fixed_positive_first": "Positive-first",
    "half_then_full_forward": "Half then full",
    "fixed_negative_first": "Negative-first",
    "multisine": "Multisine",
    "random_piecewise": "Random piecewise",
}
COLORS = dict(zip(PROBES, plt.get_cmap("tab10").colors[:len(PROBES)]))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def interval_success(prediction: np.ndarray, targets: dict, probe: str) -> float:
    return float(np.mean((prediction >= targets[f"{probe}__interval_low"])
                         & (prediction <= targets[f"{probe}__interval_high"])))


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    uncertainty = json.loads((HERE / "uncertainty.json").read_text())
    posthoc = load_npz(HERE / "predictions.npz")
    structured = load_npz(HERE / "structured_physical_predictions.npz")
    source = HERE.parent / "E025_decision_ambiguity"
    primary = load_npz(source / "audit/predictions_pre_evaluation.npz")
    targets = load_npz(source / "audit/test_targets.npz")
    selection = json.loads((source / "audit/selection.json").read_text())

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5,
                         "axes.labelsize": 8.5, "legend.fontsize": 7.5})
    output = HERE / "figures"
    output.mkdir(exist_ok=True)

    sensitivity, axes = plt.subplots(1, 2, figsize=(7.05, 2.85), constrained_layout=True)
    ax = axes[0]
    tolerance = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20])
    for probe in PROBES:
        values = results["tolerance_sweep"][f"{probe}__primary"][
            "completion_by_tolerance_m"]
        ax.plot(tolerance, [100 * values[str(value)] for value in tolerance],
                marker="o", ms=3, lw=1.4, color=COLORS[probe], label=LABELS[probe])
    ax.axvline(0.06, color="0.35", ls="--", lw=0.9)
    ax.text(0.062, 4, "registered", rotation=90, va="bottom", color="0.35")
    ax.set(title="(a) Physical tolerance audit", xlabel="Completion tolerance (m)",
           ylabel="Completion (%)", ylim=(0, 72))
    ax.legend(ncol=2, frameon=False, loc="upper left")

    ax = axes[1]
    sample_counts = np.array([9, 17, 41, 81, 401])
    positions = np.arange(len(sample_counts))
    for probe in ("fixed_positive_first", "multisine"):
        values = results["sampling_rate"][probe]
        ax.plot(positions, [100 * values[str(count)]["test_completion"]
                           for count in sample_counts], marker="o", ms=4,
                lw=1.5, color=COLORS[probe], label=LABELS[probe])
        for position, count in zip(positions, sample_counts):
            method = values[str(count)]["selected_method"].split("_", 1)[0]
            vertical = 6 if probe == "fixed_positive_first" else -10
            ax.annotate(method, (position, 100 * values[str(count)]["test_completion"]),
                        xytext=(0, vertical), textcoords="offset points", ha="center", fontsize=6.5)
    ax.set_xticks(positions, [str(value) for value in sample_counts])
    ax.set(title="(b) Sampling rate and model selection",
           xlabel="Observations from the same trace", ylabel="Completion (%)", ylim=(10, 31))
    ax.legend(frameon=False)
    sensitivity.savefig(output / "posthoc_sensitivity.pdf", bbox_inches="tight")
    sensitivity.savefig(output / "posthoc_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(sensitivity)

    attribution, axes = plt.subplots(2, 1, figsize=(7.05, 7.0), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(PROBES))
    offsets = (-0.24, -0.08, 0.08, 0.24)
    method_styles = (
        ("Primary", "o", "#2f5597"),
        ("Finite-pool posterior", "s", "#888888"),
        ("Structured", "D", "#cc6b19"),
        ("True parameters", "^", "#2b8c4b"),
    )
    for offset, (name, marker, color) in zip(offsets, method_styles):
        values = []
        for probe in PROBES:
            if name == "Primary":
                method = selection["probes"][probe]["primary_method"]
                values.append(100 * interval_success(primary[f"{probe}__{method}"], targets, probe))
            elif name == "Finite-pool posterior":
                values.append(100 * interval_success(
                    posthoc[f"{probe}__pool_bayes_calibrated"], targets, probe))
            else:
                suffix = "structured" if name == "Structured" else "true_parameter_action"
                error = structured[f"{probe}__{suffix}__landing_error"]
                values.append(100 * np.mean(error <= 0.06))
        ax.scatter(values, y + offset, s=25, marker=marker, color=color, label=name, zorder=3)
    ax.set_yticks(y, [LABELS[probe] for probe in PROBES])
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.9")
    ax.set(title="(a) Where decision accuracy is lost", xlabel="Completion at 0.06 m (%)",
           xlim=(5, 65))
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.12))

    ax = axes[1]
    entries = []
    for value in ("0.06", "0.1", "0.2"):
        item = uncertainty["tolerance_comparisons"][value]
        entries.append((f"Positive − multisine, {value} m",
                        item["positive_first_minus_multisine"]))
        entries.append((f"Negative − multisine, {value} m",
                        item["negative_first_minus_multisine"]))
    entries.append(("Positive: 401 − 9 samples",
                    uncertainty["sampling_rate_comparisons"]["fixed_positive_first"][
                        "401_minus_9_samples"]))
    entries.append(("Multisine: 17 − 9 samples",
                    uncertainty["sampling_rate_comparisons"]["multisine"][
                        "17_minus_9_samples"]))
    for probe in ("passive", "half_then_full_forward", "multisine"):
        entries.append((f"True parameters − structured: {LABELS[probe]}",
                        uncertainty["decoder_comparisons"][probe][
                            "true_parameter_action_minus_structured"]))
    yy = np.arange(len(entries))
    point = np.array([item[1]["difference_percentage_points"] for item in entries])
    intervals = np.array([item[1]["bootstrap_95_interval_percentage_points"]
                          for item in entries])
    ax.errorbar(point, yy, xerr=np.vstack((point - intervals[:, 0], intervals[:, 1] - point)),
                fmt="o", color="#333333", ecolor="#777777", ms=3.5, capsize=2, lw=1)
    ax.axvline(0, color="0.35", ls="--", lw=0.9)
    ax.set_yticks(yy, [item[0] for item in entries], fontsize=6.7)
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.92")
    ax.set(title="(b) Paired held-out comparisons",
           xlabel="Completion difference (percentage points)")

    attribution.savefig(output / "posthoc_attribution.pdf", bbox_inches="tight")
    attribution.savefig(output / "posthoc_attribution.png", dpi=240, bbox_inches="tight")
    plt.close(attribution)


if __name__ == "__main__":
    main()
