# Amendment 002: restore checkpoint library version

## Timing

This amendment was made after test inputs were generated and two prediction files were produced, but before the hidden target archive was copied, loaded, or evaluated. No completion result was available.

## Reason

The first RunPod prediction attempt loaded a checkpoint created with scikit-learn 1.6.1 into scikit-learn 1.9.0. The library emitted an incompatibility warning. That attempt is retained with its prediction lock and is excluded from the confirmatory analysis.

## Change

Prediction was repeated on the local research environment with scikit-learn 1.6.1, matching the checkpoint version. The fresh test input archive was copied without the target archive. The second prediction file and action-model checkpoint were frozen before target access and are the only inputs accepted by the evaluator.

## Unchanged elements

No source, data, seed, model, hyperparameter, hypothesis, metric, or statistical test changed. This amendment prevents a library-version difference from entering the confirmatory result.
