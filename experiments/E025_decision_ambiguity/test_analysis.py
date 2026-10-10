#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import learn


def main() -> None:
    rows = 8
    data = {"fixed_positive_first__trace": np.zeros((rows, 401, 2), dtype=np.float32)}
    first = learn.observed(data, "fixed_positive_first", "unit")
    second = learn.observed(data, "fixed_positive_first", "unit")
    assert first.shape == (rows, len(learn.TRACE_INDEX), 2)
    assert np.array_equal(first, second)
    assert np.std(first[:, :, 0]) > 0
    assert np.std(first[:, :, 1]) > np.std(first[:, :, 0])
    parameters = np.arange(rows * 8, dtype=np.float32).reshape(rows, 8)
    score = learn.d_optimal_score(first, parameters)
    assert np.isfinite(score)
    prediction = np.array([0.5, 1.5])
    assert learn.completion(prediction, np.array([0.4, 1.0]),
                            np.array([0.6, 1.4])) == 0.5
    print("E025 analysis tests passed")


if __name__ == "__main__":
    main()
