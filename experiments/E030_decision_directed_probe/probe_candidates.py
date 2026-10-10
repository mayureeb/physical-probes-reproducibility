#!/usr/bin/env python3
"""Deterministic, matched-duration candidate waveforms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "E025_decision_ambiguity"
sys.path.insert(0, str(SOURCE))
import probe_library as base  # noqa: E402

STEPS = base.STEPS
ACTIVE_STEPS = base.ACTIVE_STEPS
REST_STEPS = base.REST_STEPS
PEAK_SPEED = base.PEAK_SPEED


def finish(active: np.ndarray) -> np.ndarray:
    active = np.asarray(active, dtype=np.float64)
    active -= np.linspace(active[0], active[-1], len(active))
    peak = float(np.max(np.abs(active)))
    if peak > 0:
        active *= PEAK_SPEED / peak
    active[[0, -1]] = 0.0
    return np.concatenate((active, np.zeros(REST_STEPS))).astype(np.float32)


def chirp(start: float, end: float, phase: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, ACTIVE_STEPS)
    angle = 2 * np.pi * (start * t + 0.5 * (end - start) * t * t) + phase
    return finish(np.sin(angle) * np.sin(np.pi * t) ** 2)


def seeded_multisine(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, ACTIVE_STEPS)
    frequencies = np.asarray((1, 2, 3, 5), dtype=float)
    phases = rng.uniform(0.0, 2 * np.pi, len(frequencies))
    weights = rng.uniform(0.6, 1.4, len(frequencies))
    active = sum(w * np.sin(2 * np.pi * f * t + p)
                 for f, p, w in zip(frequencies, phases, weights))
    return finish(active * np.sin(np.pi * t) ** 2)


def seeded_piecewise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    knot_x = np.linspace(0.0, 1.0, 9)
    knot_y = rng.uniform(-1.0, 1.0, len(knot_x))
    knot_y[[0, -1]] = 0.0
    active = np.interp(np.linspace(0.0, 1.0, ACTIVE_STEPS), knot_x, knot_y)
    return finish(active)


def library() -> dict[str, np.ndarray]:
    inherited = base.library()
    result = {
        "fixed_positive_first": inherited["fixed_positive_first"],
        "fixed_negative_first": inherited["fixed_negative_first"],
        "half_then_full_forward": inherited["half_then_full_forward"],
        "inherited_multisine": inherited["multisine"],
    }
    chirp_settings = ((0.5, 2.0), (0.5, 4.0), (1.0, 3.0), (1.0, 6.0))
    for index, (start, end) in enumerate(chirp_settings):
        result[f"chirp_{index:02d}_positive"] = chirp(start, end, 0.0)
        result[f"chirp_{index:02d}_negative"] = chirp(start, end, np.pi)
    for index in range(8):
        result[f"multisine_{index:02d}"] = seeded_multisine(30030 + index)
    for index in range(16):
        result[f"piecewise_{index:02d}"] = seeded_piecewise(30130 + index)
    base.validate(result)
    return result


if __name__ == "__main__":
    for name, value in library().items():
        print(name, len(value), float(value.min()), float(value.max()))

