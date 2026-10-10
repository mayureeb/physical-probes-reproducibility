#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import probe_library as probes


def main() -> None:
    library = probes.library()
    assert set(library) == {"passive", "fixed_positive_first",
                            "half_then_full_forward", "fixed_negative_first",
                            "multisine", "random_piecewise"}
    assert np.array_equal(library["fixed_negative_first"],
                          -library["fixed_positive_first"])
    assert np.min(library["half_then_full_forward"]) >= 0.0
    assert np.max(library["half_then_full_forward"]) == np.float32(probes.PEAK_SPEED)
    assert not np.array_equal(library["multisine"], library["random_piecewise"])
    for waveform in library.values():
        assert waveform.shape == (probes.STEPS,)
        assert waveform[0] == 0.0 and waveform[-1] == 0.0
        assert np.max(np.abs(waveform)) <= probes.PEAK_SPEED + 1e-6
        assert np.all(waveform[-probes.REST_STEPS:] == 0.0)
    print("E025 probe-library tests passed")


if __name__ == "__main__":
    main()
