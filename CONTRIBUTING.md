# Contributing

Contributions should keep the project model- and dataset-agnostic. Before opening a change:

1. Add type hints and an English docstring with inputs, output, units/range, and assumptions.
2. Reject ambiguous shapes or invalid values with a clear error.
3. Add focused tests, including an analytically known case.
4. Update `docs/metrics.md` and the public imports when adding a metric.
5. Avoid dataset names, local paths, credentials, checkpoints, or preprocessing assumptions that cannot be configured explicitly.

Run `pytest` from the repository root before submitting changes.

