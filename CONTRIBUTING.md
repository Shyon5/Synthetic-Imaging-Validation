# Contributing

Contributions should be developed on a separate branch and submitted through a pull request. Contributors with write access may push a branch directly to this repository; external contributors can use a fork. Direct pushes to `main` are discouraged because they bypass review and run the test matrix only after the change has landed.

Start from an up-to-date `main` branch:

```bash
git switch main
git pull --ff-only
git switch -c add-short-metric-name
```

Keep the project model- and dataset-agnostic. Before opening a pull request:

1. Add type hints and an English docstring with inputs, output, units/range, and assumptions.
2. Reject ambiguous shapes or invalid values with a clear error.
3. Add focused tests, including an analytically known case.
4. Update `docs/metrics.md` and the public imports when adding a metric.
5. Avoid dataset names, local paths, credentials, checkpoints, or preprocessing assumptions that cannot be configured explicitly.

Install the development extras and run the same checks used in CI:

```bash
python -m pip install -e ".[test,torch,viz]"
python -m pytest --cov --cov-report=term-missing --cov-fail-under=90
```

Open a pull request into `main` and wait for the `Test suite` check before merging.
