# Amendment 001: planning batch size

**Recorded:** 2026-09-10, before opening the test-target archive.

The first target-blind planning attempt was interrupted after producing actions for
32 of 2,048 worlds. It had not written an action artifact. That batch required
about 181 seconds while the A100 had sufficient unused capacity, which projected
to an unnecessarily long run.

The implementation batch size is increased from 32 to 128 worlds. Batch size is
an execution parameter only: the physical worlds, probe traces, estimated
parameters, numerical boundary procedure, bisection count, candidate actions,
locked comparisons, and statistical tests are unchanged. The batch size can be
set through `PHYSWORLD_PLANNING_BATCH_SIZE`; the confirmatory run uses the new
default of 128. Test targets remain unopened locally and are not read by the
planning program.

The incomplete 32-world output existed only in process memory and was discarded.
No result or action file from that attempt is retained.
