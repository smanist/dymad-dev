import csv
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dymad.tuning import (
    ParameterSpec,
    TuningEvaluation,
    TuningResult,
    TuningSpec,
    batch_pattern_search_points,
    bounded_nelder_mead_search_points,
    initial_search_plan,
    plot_tuning_search,
    read_tuning_artifacts,
    tune,
    write_tuning_artifacts,
)
from dymad.utils.plot import plot_search_results


def test_parameter_spec_projects_log_int_step_parity_and_values() -> None:
    log_param = ParameterSpec("alpha", bounds=(1e-4, 1e2), scale="log")
    int_param = ParameterSpec("degree", bounds=(1, 9), value_kind="int", parity="odd")
    step_param = ParameterSpec("theta", bounds=(0.0, 1.0), step=0.25)
    values_param = ParameterSpec("stencil", values=(3, 5, 7, 9), value_kind="int")

    assert log_param.project(1e-8) == pytest.approx(1e-4)
    assert int_param.project(4) == 3
    assert step_param.project(0.62) == pytest.approx(0.5)
    assert values_param.project(6.2) == 7


def test_initial_search_plan_uses_budgeted_grid_and_random_defaults() -> None:
    grid_spec = TuningSpec(
        parameters=(
            ParameterSpec("a", bounds=(1e-2, 1e2), scale="log"),
            ParameterSpec("b", bounds=(0, 4), value_kind="int"),
        ),
        initial_budget=5,
    )
    grid_plan = initial_search_plan(grid_spec)

    random_spec = TuningSpec(
        parameters=tuple(ParameterSpec(f"p{i}", bounds=(0.0, 1.0)) for i in range(4)),
        initial_budget=6,
        seed=3,
    )
    random_plan = initial_search_plan(random_spec)

    assert grid_plan["strategy"] == "grid"
    assert len(grid_plan["candidates"]) == 5
    assert random_plan["strategy"] == "random"
    assert len(random_plan["candidates"]) == 6


def test_initial_search_plan_accepts_per_parameter_grid_budget() -> None:
    spec = TuningSpec(
        parameters=(
            ParameterSpec("a", bounds=(0.0, 1.0)),
            ParameterSpec("b", bounds=(0.0, 1.0)),
        ),
        initial_budget=(3, 4),
        initial_strategy="grid",
    )

    plan = initial_search_plan(spec)

    assert plan["initial_budget_mode"] == "per_parameter"
    assert len(plan["candidates"]) == 12
    assert len({candidate["a"] for candidate in plan["candidates"]}) == 3
    assert len({candidate["b"] for candidate in plan["candidates"]}) == 4


def test_tune_supports_parallel_initial_evaluations() -> None:
    spec = TuningSpec(
        parameters=(ParameterSpec("x", bounds=(0, 3), value_kind="int"),),
        initial_budget=4,
    )

    def evaluate(params):
        time.sleep(0.005)
        return float((params["x"] - 2) ** 2)

    result = tune(spec, evaluate, max_workers=2)

    assert result.selected_params == {"x": 2}
    assert [item.index for item in result.evaluations] == [0, 1, 2, 3]
    assert all(item.status == "ok" for item in result.evaluations)


def test_log_scale_nelder_mead_refinement_searches_in_log_coordinates() -> None:
    target = 1e-2
    spec = TuningSpec(
        parameters=(ParameterSpec("alpha", bounds=(1e-4, 1e2), scale="log"),),
        initial_budget=2,
        refinement_strategy="nelder_mead_like",
        refinement_budget=12,
    )

    def evaluate(params):
        return float((math.log10(params["alpha"]) - math.log10(target)) ** 2)

    result = tune(spec, evaluate)

    assert result.selected_params["alpha"] == pytest.approx(target, rel=1e-3)
    assert result.selected_metric < 1e-6


def test_batch_pattern_search_points_evaluates_refinement_batches() -> None:
    target = np.array([0.75, 0.25], dtype=float)
    batch_lengths: list[int] = []

    def evaluate(points):
        batch_lengths.append(len(points))
        return [float(np.sum((point - target) ** 2)) for point in points]

    evaluated = batch_pattern_search_points(
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
        evaluate_points=evaluate,
        max_evaluations=8,
        batch_size=3,
    )

    assert evaluated
    assert any(length > 1 for length in batch_lengths)
    best_point = min(evaluated, key=lambda point: np.sum((point - target) ** 2))
    assert np.linalg.norm(best_point - target) <= 0.25


def test_tune_supports_parallel_batch_pattern_search_refinement() -> None:
    spec = TuningSpec(
        parameters=(ParameterSpec("x", bounds=(0.0, 1.0)),),
        initial_budget=2,
        refinement_strategy="batch_pattern_search",
        refinement_budget=8,
    )

    def evaluate(params):
        return float((params["x"] - 0.75) ** 2)

    result = tune(spec, evaluate, max_workers=2)

    assert result.selected_params["x"] == pytest.approx(0.75, abs=0.15)
    assert any(item.phase == "refinement" for item in result.evaluations)


def test_batch_pattern_search_refinement_starts_from_log_space_best_point() -> None:
    target = {
        "bandwidth_init": 0.5623413251903494,
        "ridge_init": 1.3335214321633207e-14,
    }
    spec = TuningSpec(
        parameters=(
            ParameterSpec("bandwidth_init", bounds=(1e-4, 1e2), scale="log"),
            ParameterSpec("ridge_init", bounds=(1e-16, 1e1), scale="log"),
        ),
        initial_budget=(9, 9),
        refinement_strategy="batch_pattern_search",
        refinement_budget=2,
    )

    def evaluate(params):
        return float(
            (math.log10(params["bandwidth_init"]) - math.log10(target["bandwidth_init"])) ** 2
            + (math.log10(params["ridge_init"]) - math.log10(target["ridge_init"])) ** 2
        )

    result = tune(spec, evaluate, max_workers=2)
    refinement = [item for item in result.evaluations if item.phase == "refinement"]

    assert refinement
    assert refinement[0].params["bandwidth_init"] == pytest.approx(target["bandwidth_init"])
    assert refinement[0].params["ridge_init"] == pytest.approx(target["ridge_init"])
    assert refinement[0].params["bandwidth_init"] != pytest.approx(1e-4)
    assert refinement[0].params["ridge_init"] > 1e-15
    assert refinement[1].params["bandwidth_init"] == pytest.approx(3.1622776601683813)
    assert refinement[1].params["ridge_init"] == pytest.approx(target["ridge_init"])


def test_tune_warns_when_refinement_strategy_mismatches_worker_count() -> None:
    def evaluate(params):
        return float((params["x"] - 0.5) ** 2)

    with pytest.warns(RuntimeWarning, match="nelder_mead_like"):
        tune(
            TuningSpec(
                parameters=(ParameterSpec("x", bounds=(0.0, 1.0)),),
                initial_budget=2,
                refinement_strategy="nelder_mead_like",
                refinement_budget=1,
            ),
            evaluate,
            max_workers=2,
        )

    with pytest.warns(RuntimeWarning, match="batch_pattern_search"):
        tune(
            TuningSpec(
                parameters=(ParameterSpec("x", bounds=(0.0, 1.0)),),
                initial_budget=2,
                refinement_strategy="batch_pattern_search",
                refinement_budget=1,
            ),
            evaluate,
            max_workers=1,
        )


def test_bounded_nelder_mead_search_points_respects_bounds() -> None:
    target = np.array([0.25, 0.75], dtype=float)

    def evaluate(point: np.ndarray) -> float:
        return float(np.sum((point - target) ** 2))

    evaluated = bounded_nelder_mead_search_points(
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
        evaluate_point=evaluate,
        max_iterations=6,
    )

    assert evaluated
    assert all(np.all(point >= 0.0) and np.all(point <= 1.0) for point in evaluated)
    assert np.linalg.norm(min(evaluated, key=evaluate) - target) <= 0.25


def test_tune_records_failures_and_artifacts(tmp_path) -> None:
    spec = TuningSpec(
        parameters=(ParameterSpec("x", bounds=(0, 4), value_kind="int"),),
        initial_budget=5,
        refinement_strategy="nelder_mead_like",
        refinement_budget=2,
    )

    def evaluate(params):
        if params["x"] == 0:
            raise RuntimeError("bad candidate")
        return float((params["x"] - 2) ** 2)

    result = tune(spec, evaluate)
    write_tuning_artifacts(result, tmp_path)

    assert result.selected_params == {"x": 2}
    assert any(item.status == "failed" for item in result.evaluations)
    assert (tmp_path / "tuning_result.json").is_file()
    assert (tmp_path / "tuning_evaluations.csv").is_file()
    assert (tmp_path / "tuning_failures.csv").is_file()
    assert (tmp_path / "tuning_search.png").is_file()
    assert json.loads((tmp_path / "tuning_result.json").read_text())["selected_params"] == {"x": 2}
    with (tmp_path / "tuning_evaluations.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows


def test_read_tuning_artifacts_reconstructs_written_result(tmp_path) -> None:
    result = TuningResult(
        selected_params={"x": 2},
        selected_metric=0.0,
        evaluations=[
            TuningEvaluation({"x": 0}, "initial", 0, math.inf, "failed", 0.1, False, True, "bad"),
            TuningEvaluation({"x": 2}, "initial", 1, 0.0, "ok", 0.2),
        ],
        failures=[
            TuningEvaluation({"x": 0}, "initial", 0, math.inf, "failed", 0.1, False, True, "bad")
        ],
        candidate_plan={"strategy": "grid", "candidates": [{"x": 0}, {"x": 2}]},
        policy={"goal": "minimize"},
    )

    write_tuning_artifacts(result, tmp_path, plot=False)
    loaded = read_tuning_artifacts(tmp_path)

    assert loaded.selected_params == {"x": 2}
    assert loaded.selected_metric == pytest.approx(0.0)
    assert [item.params for item in loaded.evaluations] == [{"x": 0}, {"x": 2}]
    assert loaded.failures[0].failure_reason == "bad"
    assert math.isinf(loaded.failures[0].metric_value)
    assert loaded.candidate_plan["strategy"] == "grid"


def test_tuning_artifact_plots_2d_and_3d_searches(tmp_path) -> None:
    two_dim = TuningResult(
        selected_params={"x": 0.5, "y": 0.25},
        selected_metric=0.1,
        evaluations=[
            TuningEvaluation({"x": 0.1, "y": 0.2}, "initial", 0, 1.0, "ok", 0.0),
            TuningEvaluation({"x": 0.5, "y": 0.25}, "initial", 1, 0.1, "ok", 0.0),
        ],
        failures=[],
        candidate_plan={
            "parameter_domains": [
                {"name": "x", "scale": "linear"},
                {"name": "y", "scale": "linear"},
            ]
        },
    )
    three_dim = TuningResult(
        selected_params={"x": 0.5, "y": 0.25, "z": 2.0},
        selected_metric=0.2,
        evaluations=[
            TuningEvaluation({"x": 0.1, "y": 0.2, "z": 1.0}, "initial", 0, 1.0, "ok", 0.0),
            TuningEvaluation({"x": 0.5, "y": 0.25, "z": 2.0}, "initial", 1, 0.2, "ok", 0.0),
        ],
        failures=[],
        candidate_plan={
            "parameter_domains": [
                {"name": "x", "scale": "linear"},
                {"name": "y", "scale": "linear"},
                {"name": "z", "scale": "linear"},
            ]
        },
    )

    write_tuning_artifacts(two_dim, tmp_path / "two_dim")
    write_tuning_artifacts(three_dim, tmp_path / "three_dim")

    assert (tmp_path / "two_dim" / "tuning_search.png").is_file()
    assert (tmp_path / "three_dim" / "tuning_search.png").is_file()


def test_tuning_plot_passes_log_axis_scale_to_shared_renderer(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_plot_search_results(*args, **kwargs):
        captured["axis_scales"] = kwargs["axis_scales"]

    import dymad.utils.plot as plot_module

    monkeypatch.setattr(plot_module, "plot_search_results", fake_plot_search_results)
    result = TuningResult(
        selected_params={"alpha": 1e-2, "beta": 0.5},
        selected_metric=0.1,
        evaluations=[
            TuningEvaluation({"alpha": 1e-4, "beta": 0.0}, "initial", 0, 1.0, "ok", 0.0),
            TuningEvaluation({"alpha": 1e-2, "beta": 0.5}, "initial", 1, 0.1, "ok", 0.0),
            TuningEvaluation({"alpha": 1e2, "beta": 1.0}, "initial", 2, 2.0, "ok", 0.0),
        ],
        failures=[],
        candidate_plan={
            "parameter_domains": [
                {"name": "alpha", "scale": "log"},
                {"name": "beta", "scale": "linear"},
            ]
        },
    )

    plot_tuning_search(result, tmp_path / "ignored.png")

    assert captured["axis_scales"] == ["log", "linear"]


def test_shared_search_plot_applies_log_parameter_axis() -> None:
    fig, ax = plot_search_results(
        np.array([1e-4, 1e-2, 1.0]),
        np.array([1.0, 0.5, 0.25]),
        key_labels=["alpha"],
        metric_name="metric",
        best_idx=2,
        mode="1d",
        ifclose=False,
        axis_scales=["log"],
    )

    assert ax.get_xscale() == "log"
    plt.close(fig)
