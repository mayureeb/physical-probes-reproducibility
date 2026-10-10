# E025 amendment 002 — original-probe identity clarification

The inherited “original forward fixed probe” is not forward-only: its speed
target traverses zero to +1.2 m/s, back through zero to -1.2 m/s, and back to
zero before a final rest segment. Thus protocol items 2 and 4 would otherwise be
duplicates.

The locked identities are clarified as follows. Item 2 is the exact inherited
positive-first bidirectional waveform. Item 4 is its sign-reversed,
negative-first ordering. This creates a matched order control for the
history-dependent mechanism without changing duration, peak speed, or target
multiset. Item 3 remains nonnegative and visits half speed before full speed.

No E025 world, trace, model, validation result, or test result existed when this
clarification was made. The library still contains the seven preregistered
conditions and every scientific gate is unchanged.
