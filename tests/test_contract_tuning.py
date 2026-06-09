import csv
import json

import numpy as np
import pytest

from dymad.tuning import (
    ParameterSpec,
    TuningEvaluation,
    TuningResult,
    TuningSpec,
    bounded_nelder_mead_search_points,
    initial_search_plan,
    tune,
    write_tuning_artifacts,
)


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
