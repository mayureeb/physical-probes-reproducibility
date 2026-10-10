#!/usr/bin/env python3
"""Report post-evaluation per-parameter normalized mean squared error."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARAMETERS = ("mass", "mu", "damping", "gain", "strength", "tau1", "tau2", "mix")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=HERE / "audit/test_targets.npz",
        help="Evaluator target archive containing a parameters array.",
    )
    parser.add_argument(
        "--estimates",
        type=Path,
        default=HERE / "audit/estimates_before_targets.npz",
        help="Frozen estimate archive containing inherited and direct arrays.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "audit/per_parameter_recovery.json",
    )
    args = parser.parse_args()

    targets = np.load(args.targets)["parameters"]
    estimates = np.load(args.estimates)
    scale = targets.std(axis=0)
    result = {
        "status": "post_evaluation_parameter_recovery",
        "normalization": "test-set parameter standard deviation",
        "per_parameter_nmse": {
            method: {
                name: float(np.mean(((estimates[method][:, index] - targets[:, index])
                                     / scale[index]) ** 2))
                for index, name in enumerate(PARAMETERS)
            }
            for method in ("inherited", "direct")
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
