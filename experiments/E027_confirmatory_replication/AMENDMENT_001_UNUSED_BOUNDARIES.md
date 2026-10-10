# Amendment 001: omit unused boundary generation

## Timing

This amendment was made after two of 32 generation blocks had run and before any fresh dataset, predictions, targets, or results were written. The interrupted builder writes only after all blocks finish, so no partial artifact was available for inspection.

## Change

The generic matched-probe builder computed successful-action boundaries and oracle release outcomes for every new world. None of the seven locked hypotheses uses those quantities: all outcomes are obtained by replaying frozen predicted actions and measuring terminal landing error. The revised generator therefore stores the same clean probe traces, goals, world seeds, physical parameters, and probe-cost measurements but omits the unused boundary search and oracle release replay.

## Unchanged elements

The simulator, world count, world seed, probe definitions, observation noise, training data, frozen model choices, physical replay, hypotheses, multiplicity correction, and decision rules are unchanged. This is a computational amendment and does not respond to any observed test result.
