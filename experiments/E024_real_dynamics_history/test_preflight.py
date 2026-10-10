#!/usr/bin/env python3
"""Small deterministic tests for the locked E024 preprocessing and controls."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    return loaded


def main() -> None:
    build = module("build_data")
    learn = module("learn")
    times = np.array([2.0, 1.0, 1.0, 3.0])
    values = np.array([[20.0], [10.0], [11.0], [30.0]])
    clean_times, clean_values, duplicates = build.clean_series(times, values)
    assert duplicates == 1
    assert clean_times.tolist() == [1.0, 2.0, 3.0]
    assert clean_values[:, 0].tolist() == [11.0, 20.0, 30.0]
    history = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    runs = np.array(["a", "a", "b", "b"])
    wrong = learn.wrong_history(history, runs)
    assert np.array_equal(wrong[:, -1], history[:, -1])
    assert not np.array_equal(wrong[:, :-1], history[:, :-1])
    empty_feature = np.empty((0, 20), dtype=np.float32)
    assert empty_feature.reshape(0, build.HISTORY, 20).shape == (0, build.HISTORY, 20)
    print("E024 preflight tests passed")


if __name__ == "__main__":
    main()
