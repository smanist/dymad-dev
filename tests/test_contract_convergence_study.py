import json
import time

from dymad.studies.convergence import (
    ConvergenceStudySpec,
    MedianPlotContext,
    TuningPolicy,
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
