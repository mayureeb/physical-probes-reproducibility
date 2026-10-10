#!/usr/bin/env python3
"""Locked-length target-speed waveforms for E025 diagnostic actions."""
from __future__ import annotations

import numpy as np

DT = 0.002
STEPS = 401
ACTIVE_STEPS = 301
REST_STEPS = STEPS - ACTIVE_STEPS
PEAK_SPEED = 1.2
RANDOM_SEED = 25125
MULTISINE_SEED = 25025


def inherited_positive_first() -> np.ndarray:
    leg = np.arange(0, PEAK_SPEED + 1e-9, 8.0 * DT)
    target = np.concatenate((leg, leg[-2::-1], -leg[1:], -leg[-2::-1],
                             np.zeros(100)))
    if len(target) != STEPS:
        raise RuntimeError(f"inherited waveform changed length: {len(target)}")
    return target.astype(np.float32)


def piecewise(knots: tuple[float, ...], values: tuple[float, ...]) -> np.ndarray:
    position = np.linspace(0.0, 1.0, STEPS)
    return np.interp(position, np.asarray(knots), np.asarray(values)).astype(np.float32)


def half_then_full() -> np.ndarray:
    active = np.interp(np.linspace(0.0, 1.0, ACTIVE_STEPS),
                       (0.0, 0.18, 0.34, 0.58, 0.82, 1.0),
                       (0.0, 0.6, 0.0, 1.2, 0.0, 0.0))
    return np.concatenate((active, np.zeros(REST_STEPS))).astype(np.float32)


def multisine() -> np.ndarray:
    rng = np.random.default_rng(MULTISINE_SEED)
    time = np.linspace(0.0, 1.0, ACTIVE_STEPS)
    phases = rng.uniform(0.0, 2 * np.pi, 4)
    raw = sum(np.sin(2 * np.pi * frequency * time + phase)
              for frequency, phase in zip((1, 2, 3, 5), phases))
    envelope = np.sin(np.pi * time) ** 2
    target = raw * envelope
    target *= PEAK_SPEED / max(np.max(np.abs(target)), 1e-12)
    target[[0, -1]] = 0.0
    return np.concatenate((target, np.zeros(REST_STEPS))).astype(np.float32)


def random_piecewise() -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    knot_positions = np.linspace(0.0, 1.0, 11)
    knot_values = rng.uniform(-PEAK_SPEED, PEAK_SPEED, len(knot_positions))
    knot_values[[0, -1]] = 0.0
    target = np.interp(np.linspace(0.0, 1.0, ACTIVE_STEPS), knot_positions, knot_values)
    return np.concatenate((target, np.zeros(REST_STEPS))).astype(np.float32)


def library() -> dict[str, np.ndarray]:
    original = inherited_positive_first()
    probes = {
        "passive": np.zeros(STEPS, dtype=np.float32),
        "fixed_positive_first": original,
        "half_then_full_forward": half_then_full(),
        "fixed_negative_first": -original,
        "multisine": multisine(),
        "random_piecewise": random_piecewise(),
    }
    validate(probes)
    return probes


def validate(probes: dict[str, np.ndarray]) -> None:
    for name, target in probes.items():
        if target.shape != (STEPS,) or not np.isfinite(target).all():
            raise ValueError(f"invalid waveform: {name}")
        if abs(float(target[0])) > 1e-7 or abs(float(target[-1])) > 1e-7:
            raise ValueError(f"probe does not start and end at rest: {name}")
        if float(np.max(np.abs(target))) > PEAK_SPEED + 1e-6:
            raise ValueError(f"probe exceeds speed target cap: {name}")


if __name__ == "__main__":
    for probe_name, waveform in library().items():
        print(probe_name, len(waveform), float(np.min(waveform)),
              float(np.max(waveform)))
