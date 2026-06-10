import json
import time

import numpy as np
import pytest

from dymad.studies.convergence import (
    ArrayRegressionProblem,
    ArrayRegressionStudyConfig,
    ConvergenceStudySpec,
    CurveStyle,
    HoldoutValidationPolicy,
    KFoldValidationPolicy,
    MedianPlotContext,
    NestedArraySamples,
    NestedResamplingPolicy,
    TrainValidCountPolicy,
    TuningEvaluationContext,
    TuningPolicy,
    build_nested_trial_sample_plan,
    plot_convergence_summary,
    run_array_regression_study,
    run_convergence_study,
)
from dymad.tuning import ParameterSpec, TuningSpec


def test_convergence_study_runs_fixed_policy_and_writes_artifacts(tmp_path) -> None:
    spec = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1, 2, 4),
        trials=(0, 1),
        metrics=("error",),
        tuning_policy=TuningPolicy(mode="none", fixed_params={"m": {"alpha": 1.0}}),
        fit_window=(1, 2, 4),
        artifact_dir=tmp_path,
    )

    def evaluate(context):
        assert context.params == {"alpha": 1.0}
        return {"error": 1.0 / float(context.refinement)}

    result = run_convergence_study(spec, evaluate)

    assert len(result.raw_rows) == 6
    assert result.convergence_rates[0]["status"] == "ok"
    assert (tmp_path / "raw_results.csv").is_file()
    assert (tmp_path / "convergence_rates.json").is_file()
    assert (tmp_path / "diagnostics.json").is_file()
    assert (tmp_path / "tuning" / "m__none" / "tuning_result.json").is_file()


def test_array_regression_adapter_and_summary_plot_are_reusable(tmp_path) -> None:
    def sample(n_samples: int, rng: np.random.Generator) -> np.ndarray:
        return rng.random((n_samples, 1))

    def target(points: np.ndarray) -> np.ndarray:
        return points[:, :1]

    def score(method, split, params, include_test):
        val_pred = np.full_like(split.y_val, float(params.get("bias", 0.0)))
        row = {"validation_normalized_rmse": float(np.sqrt(np.mean((split.y_val - val_pred) ** 2)))}
        if include_test:
            test_pred = np.full_like(split.y_test, float(params.get("bias", 0.0)))
            row["error"] = float(np.sqrt(np.mean((split.y_test - test_pred) ** 2)))
        return row

    def score_folds(method, samples: NestedArraySamples, plan, trial, params):
        rows = [
            score(method, samples.split_for_fold(trial, fold), params, include_test=False)
            for fold in plan.validation_folds
        ]
        return {
            "validation_normalized_rmse": float(
                np.mean([row["validation_normalized_rmse"] for row in rows])
            )
        }

    def tuning_spec(metric_name, initial_budget, refinement_budget, refinement_strategy):
        return TuningSpec(
            parameters=(ParameterSpec("bias", bounds=(0.0, 1.0)),),
            metric_name=metric_name,
            initial_budget=1,
        )

    problem = ArrayRegressionProblem(
        name="toy",
        methods=("constant",),
        sample=sample,
        target=target,
        fit_and_score=score,
        fit_and_score_folds=score_folds,
        tuning_spec=tuning_spec,
        metrics=("error",),
        primary_metric="error",
    )
    config = ArrayRegressionStudyConfig(
        output_dir=tmp_path,
        levels=(2, 4),
        trials=2,
        n_val=2,
        n_test=3,
        initial_budget=1,
        refinement_budget=0,
        tuning_policy="none",
        plot=False,
    )

    result = run_array_regression_study(problem, config)
    plot_convergence_summary(
        result,
        tmp_path / "summary.png",
        methods=("constant",),
        center="mean",
        band="std",
        xlabel="n_train",
        ylabel="RMSE",
        styles={"constant": CurveStyle(label="Constant baseline")},
    )

    assert len(result.raw_rows) == 4
    assert (tmp_path / "summary.png").is_file()


def test_convergence_restart_evaluates_only_missing_context_results(tmp_path) -> None:
    spec = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1, 2),
        trials=2,
        metrics=("error",),
        artifact_dir=tmp_path,
    )

    def evaluate(context):
        return {"error": 10.0 * float(context.trial) + float(context.refinement)}

    initial = run_convergence_study(spec, evaluate)
    removed = sorted((tmp_path / "context_results").glob("*.json"))[-1]
    removed_payload = json.loads(removed.read_text(encoding="utf-8"))
    removed.unlink()
    calls = []

    def restart_evaluate(context):
        calls.append((context.refinement, context.trial))
        return {"error": 99.0}

    restarted = run_convergence_study(spec, restart_evaluate, restart=True)

    assert len(calls) == 1
    assert calls == [(removed_payload["refinement"], removed_payload["trial"])]
    assert len(restarted.raw_rows) == 4
    preserved = [
        row
        for row in restarted.raw_rows
        if (row["refinement"], row["trial"])
        != (removed_payload["refinement"], removed_payload["trial"])
    ]
    assert [row["error"] for row in preserved] == [
        row["error"]
        for row in initial.raw_rows
        if (row["refinement"], row["trial"])
        != (removed_payload["refinement"], removed_payload["trial"])
    ]


def test_convergence_restart_extends_trials_without_rerunning_completed(tmp_path) -> None:
    base = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1, 2),
        trials=1,
        metrics=("error",),
        artifact_dir=tmp_path,
    )
    calls = []

    def evaluate(context):
        calls.append((context.refinement, context.trial))
        return {"error": float(context.refinement)}

    run_convergence_study(base, evaluate)
    extended = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1, 2),
        trials=2,
        metrics=("error",),
        artifact_dir=tmp_path,
    )
    calls.clear()

    result = run_convergence_study(extended, evaluate, restart=True)

    assert calls == [(1, 1), (2, 1)]
    assert [(row["refinement"], row["trial"]) for row in result.raw_rows] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]


def test_convergence_restart_reuses_tuning_artifact_for_new_contexts(tmp_path) -> None:
    tuning_spec = TuningSpec(
        parameters=(ParameterSpec("alpha", bounds=(0, 2), value_kind="int"),),
        initial_budget=3,
    )
    base = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1,),
        trials=1,
        metrics=("error",),
        tuning_policy=TuningPolicy(mode="per_level", specs={"m": tuning_spec}),
        artifact_dir=tmp_path,
    )

    def tune_eval(method, refinement, trial, params):
        return float((params["alpha"] - 1) ** 2)

    def evaluate(context):
        return {"error": float(context.params["alpha"])}

    run_convergence_study(base, evaluate, tuning_evaluator=tune_eval)
    extended = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1,),
        trials=2,
        metrics=("error",),
        tuning_policy=TuningPolicy(mode="per_level", specs={"m": tuning_spec}),
        artifact_dir=tmp_path,
    )

    def fail_if_tuned(method, refinement, trial, params):
        raise AssertionError("restart should load the saved per-level tuning artifact")

    result = run_convergence_study(
        extended,
        evaluate,
        tuning_evaluator=fail_if_tuned,
        restart=True,
    )

    assert [row["params"] for row in result.raw_rows] == [{"alpha": 1}, {"alpha": 1}]


def test_convergence_interrupted_during_tuning_writes_restart_anchors(tmp_path) -> None:
    tuning_spec = TuningSpec(
        parameters=(ParameterSpec("alpha", bounds=(0, 2), value_kind="int"),),
        initial_budget=3,
    )
    spec = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(8,),
        trials=1,
        metrics=("error",),
        tuning_policy=TuningPolicy(mode="per_trial", specs={"m": tuning_spec}),
        artifact_dir=tmp_path,
        resampling=NestedResamplingPolicy(
            test_size=4,
            validation=KFoldValidationPolicy(k=4),
            seed=3,
        ),
    )

    def interrupt_tuning(context: TuningEvaluationContext):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_convergence_study(
            spec,
            lambda context: {"error": 1.0},
            tuning_context_evaluator=interrupt_tuning,
        )

    assert (tmp_path / "convergence_restart.json").is_file()
    assert (tmp_path / "sample_plans.npz").is_file()
    assert not (tmp_path / "context_results").exists()


def test_convergence_restart_extends_sample_plan_binary_ordering(tmp_path) -> None:
    base = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1000,),
        trials=1,
        metrics=("error",),
        artifact_dir=tmp_path,
        resampling=NestedResamplingPolicy(
            test_size=7,
            validation=HoldoutValidationPolicy(validation_fraction=0.2),
            seed=17,
        ),
    )

    def evaluate(context):
        assert context.sample_plan is not None
        return {"error": 1.0 / len(context.sample_plan.pool_indices)}

    run_convergence_study(base, evaluate)
    with np.load(tmp_path / "sample_plans.npz") as payload:
        old_ordering = tuple(int(value) for value in next(iter(payload.values())))

    extended = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1000, 1200),
        trials=1,
        metrics=("error",),
        artifact_dir=tmp_path,
        resampling=NestedResamplingPolicy(
            test_size=7,
            validation=HoldoutValidationPolicy(validation_fraction=0.2),
            seed=17,
        ),
    )

    result = run_convergence_study(extended, evaluate, restart=True)

    with np.load(tmp_path / "sample_plans.npz") as payload:
        new_ordering = tuple(int(value) for value in next(iter(payload.values())))
    manifest_text = (tmp_path / "convergence_restart.json").read_text(encoding="utf-8")

    assert new_ordering[: len(old_ordering)] == old_ordering
    assert len(new_ordering) >= 1200
    assert len(manifest_text) < 1200
    assert len(result.raw_rows) == 2


def test_convergence_restart_rejects_incompatible_metrics(tmp_path) -> None:
    spec = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1,),
        trials=1,
        metrics=("error",),
        artifact_dir=tmp_path,
    )
    run_convergence_study(spec, lambda context: {"error": 1.0})
    incompatible = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1,),
        trials=1,
        metrics=("loss",),
        artifact_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="not compatible"):
        run_convergence_study(incompatible, lambda context: {"loss": 1.0}, restart=True)


def test_convergence_parallel_restart_writes_context_results_and_ordered_csv(tmp_path) -> None:
    spec = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1, 2),
        trials=2,
        metrics=("error",),
        artifact_dir=tmp_path,
    )

    def evaluate(context):
        time.sleep(0.005)
        return {"error": 10.0 * float(context.refinement) + float(context.trial)}

    result = run_convergence_study(spec, evaluate, max_workers=2)

    assert len(list((tmp_path / "context_results").glob("*.json"))) == 4
    assert [(row["refinement"], row["trial"]) for row in result.raw_rows] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    raw_text = (tmp_path / "raw_results.csv").read_text(encoding="utf-8")
    assert raw_text.index("1,0") < raw_text.index("2,0")


def test_convergence_study_accepts_trial_counts() -> None:
    uniform = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(8, 16),
        trials=2,
        metrics=("error",),
    )
    per_level = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(8, 16),
        trials=(1, 3),
        metrics=("error",),
    )

    def evaluate(context):
        return {"error": float(context.trial) + 1.0 / float(context.refinement)}

    uniform_result = run_convergence_study(uniform, evaluate)
    per_level_result = run_convergence_study(per_level, evaluate)

    assert [(row["refinement"], row["trial"]) for row in uniform_result.raw_rows] == [
        (8, 0),
        (8, 1),
        (16, 0),
        (16, 1),
    ]
    assert [(row["refinement"], row["trial"]) for row in per_level_result.raw_rows] == [
        (8, 0),
        (16, 0),
        (16, 1),
        (16, 2),
    ]


def test_nested_resampling_fixed_test_nested_levels_holdout_and_kfold() -> None:
    holdout_policy = NestedResamplingPolicy(
        test_size=5,
        validation=HoldoutValidationPolicy(validation_fraction=0.25),
        seed=11,
        dev_pool_size=36,
    )
    holdout = build_nested_trial_sample_plan(
        holdout_policy,
        refinement_levels=(8, 12),
        trial=0,
    )

    assert holdout.test_indices == tuple(range(5))
    assert len(holdout.dev_ordering) == 36
    assert holdout.levels[8].pool_indices == holdout.dev_ordering[:8]
    assert holdout.levels[12].pool_indices[:8] == holdout.levels[8].pool_indices
    assert holdout.levels[8].refit_indices == holdout.levels[8].pool_indices
    fold = holdout.levels[8].validation_folds[0]
    assert len(fold.train_indices) == 6
    assert len(fold.validation_indices) == 2
    assert set(fold.train_indices).isdisjoint(fold.validation_indices)

    kfold_policy = NestedResamplingPolicy(
        test_size=5,
        validation=KFoldValidationPolicy(k=4),
        seed=11,
    )
    kfold = build_nested_trial_sample_plan(kfold_policy, refinement_levels=(8,), trial=0)
    validation_indices = [
        index for fold in kfold.levels[8].validation_folds for index in fold.validation_indices
    ]

    assert len(kfold.levels[8].validation_folds) == 4
    assert sorted(validation_indices) == sorted(kfold.levels[8].pool_indices)


def test_nested_resampling_train_valid_count_keeps_level_as_exact_train_count() -> None:
    proportional = build_nested_trial_sample_plan(
        NestedResamplingPolicy(
            test_size=5,
            validation=TrainValidCountPolicy(validation_fraction=0.25),
            seed=11,
            dev_pool_size=36,
        ),
        refinement_levels=(8, 12),
        trial=0,
    )
    level = proportional.levels[8]
    fold = level.validation_folds[0]

    assert len(proportional.dev_ordering) == 36
    assert len(level.pool_indices) == 10
    assert len(fold.train_indices) == 8
    assert len(fold.validation_indices) == 2
    assert level.refit_indices == fold.train_indices
    assert proportional.levels[12].pool_indices[:10] == level.pool_indices

    fixed = build_nested_trial_sample_plan(
        NestedResamplingPolicy(
            test_size=5,
            validation=TrainValidCountPolicy(validation_size=3),
            seed=11,
            dev_pool_size=36,
        ),
        refinement_levels=(8,),
        trial=0,
    )
    fixed_fold = fixed.levels[8].validation_folds[0]

    assert len(fixed.levels[8].pool_indices) == 11
    assert len(fixed_fold.train_indices) == 8
    assert len(fixed_fold.validation_indices) == 3
    assert fixed.levels[8].refit_indices == fixed_fold.train_indices


def test_convergence_study_passes_nested_sample_plan_to_tuning_and_final_eval() -> None:
    tuning_spec = TuningSpec(
        parameters=(ParameterSpec("alpha", values=(1.0, 2.0)),),
        initial_budget=2,
    )
    spec = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(8, 12),
        trials=2,
        metrics=("error",),
        tuning_policy=TuningPolicy(mode="per_trial", specs={"m": tuning_spec}),
        resampling=NestedResamplingPolicy(
            test_size=4,
            validation=KFoldValidationPolicy(k=4),
            seed=3,
        ),
    )
    tuning_pool_lengths = []
    final_pool_lengths = []

    def tune_eval(context: TuningEvaluationContext):
        assert context.sample_plan is not None
        tuning_pool_lengths.append(len(context.sample_plan.pool_indices))
        return {
            "metric": float(context.params["alpha"]),
            "std_metric": float(len(context.sample_plan.validation_folds)),
        }

    def evaluate(context):
        assert context.sample_plan is not None
        final_pool_lengths.append(len(context.sample_plan.refit_indices))
        assert context.sample_plan.refit_indices == context.sample_plan.pool_indices
        return {"error": 1.0 / len(context.sample_plan.refit_indices)}

    result = run_convergence_study(spec, evaluate, tuning_context_evaluator=tune_eval)

    assert sorted(set(tuning_pool_lengths)) == [8, 12]
    assert final_pool_lengths == [8, 8, 12, 12]
    assert result.trial_statistics[0]["median"] > 0.0
    assert {"stderr", "q05", "q25", "q75", "q95"} <= set(result.trial_statistics[0])


def test_convergence_study_per_trial_tuning_saves_each_tuning(tmp_path) -> None:
    tuning_spec = TuningSpec(
        parameters=(ParameterSpec("alpha", bounds=(0, 4), value_kind="int"),),
        initial_budget=5,
    )
    spec = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1, 2),
        trials=(0, 1),
        metrics=("error",),
        tuning_policy=TuningPolicy(mode="per_trial", specs={"m": tuning_spec}),
        artifact_dir=tmp_path,
    )

    def tune_eval(method, refinement, trial, params):
        return float((params["alpha"] - (1 + int(trial))) ** 2)

    def evaluate(context):
        return {"error": 1.0 / (float(context.refinement) + float(context.params["alpha"]))}

    result = run_convergence_study(spec, evaluate, tuning_evaluator=tune_eval)

    assert len(result.tuning_results) == 4
    assert len(list((tmp_path / "tuning").glob("*/tuning_result.json"))) == 4


def test_convergence_study_supports_parallel_workers(tmp_path) -> None:
    tuning_spec = TuningSpec(
        parameters=(ParameterSpec("alpha", bounds=(0, 3), value_kind="int"),),
        initial_budget=4,
    )
    spec = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1, 2),
        trials=2,
        metrics=("error",),
        tuning_policy=TuningPolicy(mode="per_trial", specs={"m": tuning_spec}),
        artifact_dir=tmp_path,
    )

    def tune_eval(method, refinement, trial, params):
        time.sleep(0.005)
        return float((params["alpha"] - int(trial)) ** 2)

    def evaluate(context):
        time.sleep(0.005)
        return {"error": float(context.params["alpha"]) + 1.0 / float(context.refinement)}

    result = run_convergence_study(
        spec,
        evaluate,
        tuning_evaluator=tune_eval,
        max_workers=2,
        tuning_max_workers=2,
    )

    assert [(row["refinement"], row["trial"]) for row in result.raw_rows] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    assert len(result.tuning_results) == 4


def test_convergence_study_writes_supplied_median_prediction_plots(tmp_path) -> None:
    spec = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1, 2),
        trials=(0, 1, 2),
        metrics=("error",),
        tuning_policy=TuningPolicy(mode="none", fixed_params={"m": {"alpha": 1.0}}),
        artifact_dir=tmp_path,
    )
    plotted: list[tuple[str, int, int]] = []

    def evaluate(context):
        trial_factor = {0: 2.0, 1: 1.0, 2: 3.0}[int(context.trial)]
        return {"error": trial_factor / float(context.refinement)}

    def plotter(context: MedianPlotContext) -> None:
        plotted.append((context.method, int(context.refinement), int(context.trial)))
        context.output_path.write_bytes(b"plot")

    result = run_convergence_study(spec, evaluate, median_plotter=plotter)

    assert plotted == [("m", 1, 0), ("m", 2, 0)]
    assert result.median_plot_paths == {
        "m__level_1": "median_predictions/m__level_1.png",
        "m__level_2": "median_predictions/m__level_2.png",
    }
    assert (tmp_path / "median_prediction_plots.json").is_file()
    assert (tmp_path / "median_predictions" / "m__level_1.png").is_file()


def test_convergence_study_per_level_and_reference_level_reuse(tmp_path) -> None:
    tuning_spec = TuningSpec(
        parameters=(ParameterSpec("alpha", bounds=(0, 4), value_kind="int"),),
        initial_budget=5,
    )
    calls = []

    def tune_eval(method, refinement, trial, params):
        calls.append((method, refinement, trial, params["alpha"]))
        return float((params["alpha"] - int(float(refinement))) ** 2)

    def evaluate(context):
        return {"error": 1.0}

    per_level = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1, 2),
        trials=(0, 1),
        metrics=("error",),
        tuning_policy=TuningPolicy(mode="per_level", specs={"m": tuning_spec}),
    )
    run_convergence_study(per_level, evaluate, tuning_evaluator=tune_eval)
    per_level_call_count = len(calls)

    calls.clear()
    reference = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1, 2),
        trials=(0, 1),
        metrics=("error",),
        tuning_policy=TuningPolicy(
            mode="reference_level", specs={"m": tuning_spec}, reference_level=2
        ),
    )
    run_convergence_study(reference, evaluate, tuning_evaluator=tune_eval)

    assert per_level_call_count == 10
    assert len(calls) == 5


def test_convergence_diagnostics_flag_nonmonotone_boundary_and_variance(tmp_path) -> None:
    tuning_spec = TuningSpec(
        parameters=(ParameterSpec("alpha", bounds=(0, 2), value_kind="int"),),
        initial_budget=3,
    )
    spec = ConvergenceStudySpec(
        methods=("m",),
        refinement_levels=(1, 2, 4),
        trials=(0, 1),
        metrics=("error",),
        tuning_policy=TuningPolicy(mode="per_level", specs={"m": tuning_spec}),
        fit_window=(1, 2, 4),
        high_variance_cv_threshold=0.1,
        artifact_dir=tmp_path,
    )

    def tune_eval(method, refinement, trial, params):
        return float(params["alpha"])

    def evaluate(context):
        base = {1: 1.0, 2: 2.0, 4: 1.5}[int(context.refinement)]
        return {"error": base if int(context.trial) == 0 else base * 2.0}

    result = run_convergence_study(spec, evaluate, tuning_evaluator=tune_eval)
    kinds = {diagnostic.kind for diagnostic in result.diagnostics}
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())

    assert "non_monotone_fit_window" in kinds
    assert "tuning_failures_or_boundary" in kinds
    assert "high_trial_variance" in kinds
    assert diagnostics
