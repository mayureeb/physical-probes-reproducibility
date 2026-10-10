#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import evaluate


def main() -> None:
    completed = evaluate.physical_completion(
        np.array([0.559, 0.57, 0.641]), np.array([0.50, 0.50, 0.70]))
    assert completed.tolist() == [True, False, True]
    print("E025 evaluation-predicate test passed")


if __name__ == "__main__":
    main()
