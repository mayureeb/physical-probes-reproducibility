# Amendment 001: remove an unused PyTorch import dependency

Recorded after fresh data generation began and before validation selection or
test-target access. The first analysis source imported E025's complete learning
module only to reuse its deterministic observation-noise and flattening helper
functions. That module imports PyTorch, which is not installed in the locked MJX
generation environment.

The helpers are copied verbatim into the E026 analysis source. Their constants,
seed construction, covariance, trace indices, and outputs are unchanged. This
amendment changes no scientific design, data, selection rule, model family,
metric, threshold, or test-access order.
