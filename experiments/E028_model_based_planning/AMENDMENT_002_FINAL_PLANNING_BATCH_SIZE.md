# Amendment 002: final planning batch size

**Recorded:** 2026-09-10, before opening the test-target archive.

The 128-world timing run was interrupted after two batches (256 of 2,048
worlds). It had not written an action artifact. The batches took 236 and 205
seconds, respectively, and the A100 remained able to support a larger batch.

The default execution batch is therefore increased from 128 to 256 worlds for
the complete run. As in Amendment 001, this changes only how independent worlds
are grouped for computation. It does not change the worlds, estimated physical
parameters, probes, simulator, bisection procedure, actions, hypotheses, or
statistical tests. The incomplete in-memory actions were discarded. Test targets
remain unopened locally and are not read by the planning program.
