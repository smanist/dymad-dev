---
name: dymad-tuning-convergence-study
description: Build concrete DyMAD tuning/convergence studies that use package-owned tuning and convergence primitives, create study-specific problem modules and reproducible CLI wrappers, and add workflow smoke tests. Use when a user asks an agent to implement, adapt, or review a freeform convergence study involving custom targets, sampling, models, fitting/scoring, tuning specs, or convergence plots. Do not use for ordinary DyMAD MCP train/eval workflows.
---

# DyMAD Tuning/Convergence Study

## Overview

Use this skill to implement a concrete tuning/convergence study where DyMAD owns the reusable
study machinery but the study owns the scientific details. Prefer a package-primitive-backed
Python implementation plus a thin per-study CLI; do not add MCP exposure unless the package already
owns a named, serializable workflow.

## Required Repo Context

Before adding or changing runnable scripts under `scripts/`, read:

- `docs/developer/architecture.md`
- `docs/developer/feature-placement.md`
- `docs/developer/example-script-pattern.md`

Use the existing tuning/convergence example as the local pattern:

- `scripts/tuning_convergence/cartesian_high_freq_krr_problem.py`
- `scripts/tuning_convergence/cartesian_high_freq_krr_cli.py`
- `tests/test_workflow_tuning_convergence_example.py`

## Stable Contract

Depend on public package primitives instead of copying orchestration code:

- `dymad.tuning.ParameterSpec`
- `dymad.tuning.TuningSpec`
- `dymad.studies.convergence.ArrayRegressionProblem`
- `dymad.studies.convergence.ArrayRegressionStudyConfig`
- `dymad.studies.convergence.run_array_regression_study`
- lower-level `ConvergenceStudySpec` and `run_convergence_study` only when the array-regression
  adapter is not a fit

Keep target functions, sampling, model construction, fitting, scoring, and plotting in the
study-specific implementation. Keep reusable tuning and convergence behavior in the package.

## File Shape

For a new study, default to:

```text
scripts/tuning_convergence/<study>_problem.py
scripts/tuning_convergence/<study>_cli.py
tests/test_workflow_<study>_tuning_convergence.py
```

Use separate target/data helper modules only when they reduce real complexity. Avoid adding MCP
tools, registry entries, compiler schemas, persisted handles, or executor workflows for this
freeform scaffold stage.

## Problem Module Checklist

In `<study>_problem.py`:

1. Define a `make_problem(...) -> ArrayRegressionProblem` factory.
2. Implement deterministic sampling, target generation, model construction, and scoring.
3. Implement `fit_and_score(method, split, params, include_test)` and return scalar metrics.
4. Implement `fit_and_score_folds(...)` when nested validation or k-fold tuning is supported.
5. Implement `tuning_spec(metric_name, initial_budget, refinement_budget, refinement_strategy)`.
6. Include a primary metric that exists in final evaluation rows, commonly `"error"`.
7. Add optional convergence and prediction plotters only when useful and keep them noninteractive
   with an Agg backend for testability.

Make tests cheap by ensuring the study can run with tiny levels, one trial, and a tiny tuning
budget.

## CLI Wrapper Checklist

In `<study>_cli.py`:

1. Keep argument parsing thin and explicit.
2. Convert CLI args into `ArrayRegressionStudyConfig`.
3. Call `make_problem(...)`.
4. Call `run_array_regression_study(problem, config, make_plot=...)`.
5. Print `Wrote convergence artifacts to ...` on success.
6. Include switches for small reproducible runs: `--workdir`, `--levels`, `--trials`,
   `--initial-budget`, `--refinement-budget`, `--seed`, `--max-workers`, `--no-plot`, and
   `--no-prediction-plots` when prediction plots exist.
7. Support restart or nested validation options only when the study needs them.

Do not make the package-level `dymad` CLI generic for arbitrary freeform study code unless the
workflow has become a stable package-owned capability.

## Workflow Test Checklist

Add a focused pytest file named with an allowed workflow prefix:

```text
tests/test_workflow_<study>_tuning_convergence.py
```

The test should cover CLI reproducibility, study wiring, and artifact layout. It should not assert
production convergence rates or full scientific accuracy.

Run the CLI through `subprocess.run` with tiny settings, for example two levels, one trial,
`--initial-budget 2`, `--refinement-budget 0`, `--max-workers 1`, and plotting disabled unless
plot generation is the behavior under test. Assert:

- the CLI exits successfully
- stdout contains `Wrote convergence artifacts`
- `raw_results.csv` exists
- `convergence_summary.csv` exists
- `convergence_rates.json` exists
- `tuning/*/tuning_result.json` exists when tuning is enabled
- `tuning/*/tuning_evaluations.csv` exists when tuning is enabled
- plot files exist only in tests that enable plotting

If the study exposes reusable target functions or helpers, add small deterministic assertions for
shape and finite values in the same workflow test file or the closest existing test.

## Verification

After editing Python files, run the nearest targeted pytest coverage first, then the repo-required
static checks:

```bash
pytest tests/test_workflow_<study>_tuning_convergence.py
make check
```

If a broader existing tuning/convergence package behavior changed, also run:

```bash
pytest tests/test_contract_tuning.py tests/test_contract_convergence_study.py
```
