from __future__ import annotations

import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cartesian_high_freq_krr_cli import (
    METHODS,
    NestedCartesianSamples,
    fit_and_score,
    fit_and_score_folds,
    make_plot,
    make_split,
    plot_truth_vs_prediction,
    tuning_spec,
)

from dymad.studies.convergence import (
    ConvergenceEvaluationContext,
    ConvergenceStudySpec,
    HoldoutValidationPolicy,
    KFoldValidationPolicy,
    MedianPlotContext,
    NestedResamplingPolicy,
    TrainValidCountPolicy,
    TuningEvaluationContext,
    TuningPolicy,
    run_convergence_study,
)


# fmt: off
OUTPUT_DIR = Path("./runs")
LEVELS = (512, 1024, 2048, 4096, 8192)
TRIALS = 5
N_VAL = 1024
N_TEST = 4096
INITIAL_BUDGET = (9, 9)
REFINEMENT_STRATEGY = "batch_pattern_search"
REFINEMENT_BUDGET = 64 if REFINEMENT_STRATEGY == "batch_pattern_search" else 20
TUNING_POLICY = "per_trial"
SEED = 0
MAX_WORKERS = 4
RESAMPLING_MODE = "nested-fixed-test"
VALIDATION_MODE = "train-valid-count"
VALIDATION_FRACTION = 0.25
VALIDATION_SIZE = 1024
K_FOLDS = 4
POOL_MULTIPLIER = 2
CONFIDENCE_BAND = None

RESTART = True

ifrun = 1
ifplt = 1
ifprd = 1
# fmt: on


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
split_cache: dict[tuple[int, int], Any] = {}
nested_policy: NestedResamplingPolicy | None = None
nested_samples: NestedCartesianSamples | None = None
result: Any | None = None


def split_for(refinement: int | float | str, trial: int | str) -> Any:
    key = (int(refinement), int(trial))
    if key not in split_cache:
        split_cache[key] = make_split(int(refinement), N_VAL, N_TEST, int(trial))
    return split_cache[key]


def split_for_context(context: ConvergenceEvaluationContext | MedianPlotContext) -> Any:
    if nested_samples is None or context.sample_plan is None:
        return split_for(context.refinement, context.trial)
    return nested_samples.split_for_refit(context.trial, context.sample_plan)


def tune_eval(method: str, refinement: int | float | str, trial: int | str, params: dict[str, Any]):
    split = split_for(refinement, trial)
    return fit_and_score(
        method,
        split,
        float(params["bandwidth_init"]),
        float(params["ridge_init"]),
        include_test=False,
    )


def tune_context_eval(context: TuningEvaluationContext):
    if nested_samples is None or context.sample_plan is None:
        return tune_eval(context.method, context.refinement, context.trial, context.params)
    return fit_and_score_folds(
        context.method,
        nested_samples,
        context.sample_plan,
        context.trial,
        float(context.params["bandwidth_init"]),
        float(context.params["ridge_init"]),
    )


def study_eval(context: ConvergenceEvaluationContext) -> dict[str, Any]:
    split = split_for_context(context)
    return fit_and_score(
        context.method,
        split,
        float(context.params["bandwidth_init"]),
        float(context.params["ridge_init"]),
        include_test=True,
    )


def median_plotter(context: MedianPlotContext) -> None:
    plot_truth_vs_prediction(context, split_for_context(context))


def nested_resampling_policy() -> NestedResamplingPolicy | None:
    if RESAMPLING_MODE != "nested-fixed-test":
        return None
    if VALIDATION_MODE == "holdout":
        validation = HoldoutValidationPolicy(validation_fraction=VALIDATION_FRACTION)
    elif VALIDATION_MODE == "kfold":
        validation = KFoldValidationPolicy(k=K_FOLDS)
    else:
        validation = TrainValidCountPolicy(
            validation_fraction=None if VALIDATION_SIZE is not None else VALIDATION_FRACTION,
            validation_size=VALIDATION_SIZE,
        )
    return NestedResamplingPolicy(
        test_size=N_TEST,
        validation=validation,
        seed=SEED,
        dev_pool_size=max(max(LEVELS) * POOL_MULTIPLIER, max(LEVELS) + VALIDATION_SIZE),
    )


if ifrun:
    nested_policy = nested_resampling_policy()
    if nested_policy is not None:
        nested_samples = NestedCartesianSamples(
            max_train=nested_policy.dev_pool_size or max(LEVELS),
            n_test=N_TEST,
            seed=SEED,
            trials=tuple(range(TRIALS)) if isinstance(TRIALS, int) else TRIALS,
        )
    specs = {
        method: replace(
            tuning_spec("validation_normalized_rmse", INITIAL_BUDGET, REFINEMENT_BUDGET),
            refinement_strategy=REFINEMENT_STRATEGY if REFINEMENT_BUDGET > 0 else None,
        )
        for method in METHODS
    }
    study_spec = ConvergenceStudySpec(
        methods=METHODS,
        refinement_levels=LEVELS,
        trials=TRIALS,
        metrics=("error", "test_physical_rmse", "test_normalized_max_abs", "fit_seconds"),
        tuning_policy=TuningPolicy(mode=TUNING_POLICY, specs=specs),
        fit_window=LEVELS,
        artifact_dir=OUTPUT_DIR,
        primary_metric="error",
        resampling=nested_policy,
    )
    result = run_convergence_study(
        study_spec,
        study_eval,
        tuning_evaluator=tune_eval if nested_policy is None else None,
        tuning_context_evaluator=tune_context_eval if nested_policy is not None else None,
        median_plotter=median_plotter if ifprd else None,
        max_workers=MAX_WORKERS,
        tuning_max_workers=MAX_WORKERS,
        restart=RESTART,
    )
    print(f"Wrote convergence artifacts to {OUTPUT_DIR}")
    if result.diagnostics:
        print(f"Diagnostics: {len(result.diagnostics)} advisory item(s); see diagnostics.json")

if ifplt:
    if result is None:
        raise RuntimeError("Set ifrun=1 before plotting so the study result is available.")
    plot_center = "median" if nested_policy is not None else "mean"
    plot_band = CONFIDENCE_BAND or ("iqr" if nested_policy is not None else "std")
    make_plot(result, OUTPUT_DIR, center=plot_center, band=plot_band)
    print(f"Wrote plot to {OUTPUT_DIR / 'convergence.png'}")
