from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dymad.core.transform_module import TransformModule
from dymad.io import Split, TransformSpec
from dymad.studies.convergence.core import (
    ConvergenceEvaluationContext,
    ConvergenceStudyResult,
    ConvergenceStudySpec,
    MedianPlotContext,
    TuningEvaluationContext,
    TuningPolicy,
    run_convergence_study,
)
from dymad.studies.convergence.resampling import (
    HoldoutValidationPolicy,
    KFoldValidationPolicy,
    LevelSamplePlan,
    NestedResamplingPolicy,
    TrainValidCountPolicy,
)
from dymad.tuning import TuningSpec


@dataclass(frozen=True)
class ArrayRegressionProblem:
    name: str
    methods: tuple[str, ...]
    sample: Callable[[int, np.random.Generator], np.ndarray]
    target: Callable[[np.ndarray], np.ndarray]
    fit_and_score: Callable[[str, Split, Mapping[str, Any], bool], dict[str, Any]]
    fit_and_score_folds: Callable[
        [str, NestedArraySamples, LevelSamplePlan, int | str, Mapping[str, Any]],
        dict[str, Any],
    ]
    tuning_spec: Callable[[str, int | tuple[int, ...], int, str | None], TuningSpec]
    metrics: tuple[str, ...]
    primary_metric: str
    prediction_plotter: Callable[[MedianPlotContext, Split], None] | None = None
    x_transform: TransformSpec = "std"
    y_transform: TransformSpec = "std"

    def split_from_arrays(
        self,
        *,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
    ) -> Split:
        return Split.from_arrays(
            x_train=x_train,
            y_train=self.target(x_train),
            x_val=x_val,
            y_val=self.target(x_val),
            x_test=x_test,
            y_test=self.target(x_test),
            x_transform=_copy_transform_spec(self.x_transform),
            y_transform=_copy_transform_spec(self.y_transform),
        )

    def make_split(self, n_train: int, n_val: int, n_test: int, seed: int) -> Split:
        rng = np.random.default_rng(100_000 * seed + 97 * n_train + 19_889)
        return self.split_from_arrays(
            x_train=self.sample(n_train, rng),
            x_val=self.sample(n_val, rng),
            x_test=self.sample(n_test, rng),
        )


class NestedArraySamples:
    def __init__(
        self,
        problem: ArrayRegressionProblem,
        *,
        max_train: int,
        n_test: int,
        seed: int,
        trials: tuple[int | str, ...],
    ):
        self._problem = problem
        test_rng = np.random.default_rng(1_000_000_007 + seed)
        self.x_test = problem.sample(n_test, test_rng)
        self.x_dev_by_trial = {
            trial: problem.sample(max_train, np.random.default_rng(2_000_000_011 + seed + index))
            for index, trial in enumerate(trials)
        }

    def split_for_fold(self, trial: int | str, fold: Any) -> Split:
        x_dev = self.x_dev_by_trial[trial]
        return self._problem.split_from_arrays(
            x_train=x_dev[list(fold.train_indices)],
            x_val=x_dev[list(fold.validation_indices)],
            x_test=self.x_test,
        )

    def split_for_refit(self, trial: int | str, plan: LevelSamplePlan) -> Split:
        x_dev = self.x_dev_by_trial[trial]
        x_train = x_dev[list(plan.refit_indices)]
        return self._problem.split_from_arrays(
            x_train=x_train,
            x_val=x_train[:1],
            x_test=self.x_test,
        )


def _copy_transform_spec(transform: TransformSpec) -> TransformSpec:
    if isinstance(transform, TransformModule):
        return deepcopy(transform)
    return transform


@dataclass(frozen=True)
class ArrayRegressionStudyConfig:
    output_dir: str | Path
    levels: tuple[int, ...]
    trials: int | tuple[int | str, ...]
    n_val: int
    n_test: int
    initial_budget: int | tuple[int, ...]
    refinement_budget: int
    refinement_strategy: str | None = None
    tuning_policy: str = "per_trial"
    seed: int = 0
    max_workers: int = 1
    resampling_mode: str = "legacy"
    validation_mode: str = "holdout"
    validation_fraction: float = 0.25
    validation_size: int | None = None
    k_folds: int = 4
    pool_multiplier: int = 1
    confidence_band: str | None = None
    restart: bool = False
    plot: bool = True
    prediction_plots: bool = True


def run_array_regression_study(
    problem: ArrayRegressionProblem,
    config: ArrayRegressionStudyConfig,
    *,
    make_plot: Callable[[ConvergenceStudyResult, Path, str, str], None] | None = None,
) -> ConvergenceStudyResult:
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    nested_policy = _nested_resampling_policy(config)
    nested_samples = (
        NestedArraySamples(
            problem,
            max_train=nested_policy.dev_pool_size or max(config.levels),
            n_test=config.n_test,
            seed=config.seed,
            trials=_trial_ids_for_all_levels(config.trials, len(config.levels)),
        )
        if nested_policy is not None
        else None
    )
    split_cache: dict[tuple[int, int], Split] = {}

    def split_for(refinement: int | float | str, trial: int | str) -> Split:
        key = (int(refinement), int(trial))
        if key not in split_cache:
            split_cache[key] = problem.make_split(
                int(refinement), config.n_val, config.n_test, int(trial)
            )
        return split_cache[key]

    def split_for_context(context: ConvergenceEvaluationContext | MedianPlotContext) -> Split:
        if nested_samples is None or context.sample_plan is None:
            return split_for(context.refinement, context.trial)
        return nested_samples.split_for_refit(context.trial, context.sample_plan)

    def tune_eval(
        method: str, refinement: int | float | str, trial: int | str, params: dict[str, Any]
    ) -> dict[str, Any]:
        return problem.fit_and_score(method, split_for(refinement, trial), params, False)

    def tune_context_eval(context: TuningEvaluationContext) -> dict[str, Any]:
        if nested_samples is None or context.sample_plan is None:
            return tune_eval(context.method, context.refinement, context.trial, context.params)
        return problem.fit_and_score_folds(
            context.method,
            nested_samples,
            context.sample_plan,
            context.trial,
            context.params,
        )

    def study_eval(context: ConvergenceEvaluationContext) -> dict[str, Any]:
        return problem.fit_and_score(
            context.method, split_for_context(context), context.params, True
        )

    def median_plotter(context: MedianPlotContext) -> None:
        if problem.prediction_plotter is None:
            return
        problem.prediction_plotter(context, split_for_context(context))

    specs = {
        method: problem.tuning_spec(
            "validation_normalized_rmse",
            config.initial_budget,
            config.refinement_budget,
            config.refinement_strategy,
        )
        for method in problem.methods
    }
    study_spec = ConvergenceStudySpec(
        methods=problem.methods,
        refinement_levels=config.levels,
        trials=config.trials,
        metrics=problem.metrics,
        tuning_policy=TuningPolicy(mode=config.tuning_policy, specs=specs),
        fit_window=config.levels,
        artifact_dir=output_dir,
        primary_metric=problem.primary_metric,
        resampling=nested_policy,
    )
    result = run_convergence_study(
        study_spec,
        study_eval,
        tuning_evaluator=tune_eval if nested_policy is None else None,
        tuning_context_evaluator=tune_context_eval if nested_policy is not None else None,
        median_plotter=median_plotter
        if config.prediction_plots and problem.prediction_plotter is not None
        else None,
        max_workers=config.max_workers,
        tuning_max_workers=config.max_workers,
        restart=config.restart,
    )
    if config.plot and make_plot is not None:
        center = "median" if nested_policy is not None else "mean"
        band = config.confidence_band or ("iqr" if nested_policy is not None else "std")
        make_plot(result, output_dir, center, band)
    return result


def _nested_resampling_policy(config: ArrayRegressionStudyConfig) -> NestedResamplingPolicy | None:
    if config.resampling_mode != "nested-fixed-test":
        return None
    if config.validation_mode == "holdout":
        validation = HoldoutValidationPolicy(validation_fraction=config.validation_fraction)
    elif config.validation_mode == "kfold":
        validation = KFoldValidationPolicy(k=config.k_folds)
    elif config.validation_mode == "train-valid-count":
        validation = TrainValidCountPolicy(
            validation_fraction=None
            if config.validation_size is not None
            else config.validation_fraction,
            validation_size=config.validation_size,
        )
    else:
        raise ValueError(f"unknown validation mode {config.validation_mode!r}")
    max_level = max(config.levels)
    validation_count = (
        int(config.validation_size)
        if config.validation_size is not None
        else max(1, int(round(max_level * config.validation_fraction)))
        if config.validation_mode == "train-valid-count"
        else 0
    )
    return NestedResamplingPolicy(
        test_size=config.n_test,
        validation=validation,
        seed=config.seed,
        dev_pool_size=max(max_level * config.pool_multiplier, max_level + validation_count),
    )


def _trial_ids_for_all_levels(
    trials: int | tuple[int | str, ...], n_levels: int
) -> tuple[int | str, ...]:
    if isinstance(trials, int):
        return tuple(range(trials))
    if trials and all(isinstance(item, int) and item > 0 for item in trials):
        return tuple(range(max(int(item) for item in trials[:n_levels])))
    return trials
