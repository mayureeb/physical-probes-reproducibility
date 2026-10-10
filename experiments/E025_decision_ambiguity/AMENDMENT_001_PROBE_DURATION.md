# E025 amendment 001 — probe duration correction

The registered protocol incorrectly described the original fixed probe as
lasting 1.2 seconds. In the inherited E020/E021 simulator, **1.2 is the peak
target speed in metres per second**. The exact inherited waveform contains 401
steps at 0.002 seconds per step, so its duration is 0.802 seconds.

All E025 active probes and the passive observation interval are therefore locked
to 401 simulator steps (0.802 seconds), while retaining the common 1.2 m/s speed
limit. No E025 implementation, generated world, observation, validation result,
or test result existed when this units correction was made. No scientific gate,
seed, comparison, or downstream decision changed.
