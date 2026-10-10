# Amendment 1: remove an unused training dependency

The first remote invocation stopped during module import before candidate-world
generation began. Importing the earlier learning module solely for its
D-optimal scoring function resolved to an unrelated module and required PyTorch,
which is not installed in the locked MJX environment.

The eight-line D-optimal calculation already specified by the protocol is now
defined locally. The formula, candidate library, seeds, budgets, selection rules,
hypotheses, and sample sizes are unchanged. No candidate data or result existed
when this amendment was made.
