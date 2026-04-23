import logging

import numpy as np
import pytest
import torch
from scipy.signal import savgol_filter

import dymad.training.phases as phases_module
from dymad.core import GraphSeries, RegularSeries, RegularTrainerBatch
from dymad.training import driver
from dymad.training.execution_services import ExecutionServices
from dymad.training.helper import (
    CVResult,
    bounded_nelder_mead_search_points,
    nelder_mead_like_search_indices,
    select_best_cv_result,
)
from dymad.training.phase_runtime import (
    ArtifactRegistry,
    ModelArtifact,
    OptimizerStateArtifact,
    PhaseContext,
    PhaseRecord,
    PhaseResult,
    TrainerState,
    TrainingCheckpointError,
    TrainingHistoryArtifact,
)
from dymad.training.phases import (
    AnalysisPhaseSpec,
    DataPhaseSpec,
    ExportPhaseSpec,
    LinearSolvePhaseSpec,
    OptimizerPhaseSpec,
    PhaseSpecValidationError,
    build_phase,
    normalize_phase_specs,
)
from dymad.training.trainer_run import TrainerRun
from dymad.utils import load_config


def _build_phase_context():
    marker = object()
    return (
        PhaseContext(
            train_set=[marker],
            valid_set=[marker],
            train_loader=marker,
            valid_loader=marker,
            train_md={"dt_and_n_steps": [(0.1, 5)]},
            valid_md={"dt_and_n_steps": [(0.1, 5)]},
        ),
        marker,
    )


def _build_regular_series(offset: float = 0.0) -> RegularSeries:
    time = torch.linspace(0.0, 0.8, 9)
    state = torch.stack((torch.sin(time + offset), torch.cos(time + offset)), dim=1)
    control = torch.stack((time, time**2), dim=1)
    return RegularSeries(time=time, state=state, control=control, meta={"series": float(offset)})


def _build_graph_series(offset: float = 0.0) -> GraphSeries:
    time = torch.linspace(0.0, 0.8, 9)
    node_state = torch.stack(
        (
            torch.stack((torch.sin(time + offset), torch.cos(time + offset)), dim=1),
            torch.stack((torch.sin(time + offset + 0.1), torch.cos(time + offset + 0.1)), dim=1),
        ),
        dim=1,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0])
    return GraphSeries(
        time=time,
        node_state=node_state,
        edge_index=edge_index,
        edge_weight=edge_weight,
        meta={"graph": True},
    )


def _build_data_phase(
    config: dict,
    spec: DataPhaseSpec,
):
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    return build_phase(
        spec,
        config=config,
        model_class=object,
        dtype=torch.float32,
        execution_services=execution_services,
    )


def test_phase_result_get_metric_reads_typed_trainer_state():
    context, _ = _build_phase_context()
    trainer_state = TrainerState(config={}, best_loss={"valid_total": 0.456})
    result = PhaseResult(
        name="phase_0",
        kind="optimizer",
        trainer_state=trainer_state,
        phase_context=context,
        artifacts=ArtifactRegistry(),
    )

    assert result.get_metric("total") == 0.456


def test_safe_plot_returns_false_when_plotting_fails(caplog):
    with caplog.at_level(logging.WARNING):
        wrote_plot = phases_module._safe_plot(
            logging.getLogger("test.plot"),
            label="history plot 'demo'",
            fn=lambda: (_ for _ in ()).throw(RuntimeError("backend unavailable")),
        )

    assert wrote_plot is False
    assert "Skipping history plot 'demo' due to plotting failure." in caplog.text


def test_normalize_phase_specs_expands_repeat_schedule():
    specs = normalize_phase_specs(
        {
            "model": {"name": "demo"},
            "phases": [
                {
                    "repeat": {
                        "times": 2,
                        "phases": [
                            {
                                "type": "linear_solve",
                                "name": "ls",
                                "method": "truncated",
                                "params": 2,
                            },
                            {
                                "type": "optimizer",
                                "name": "node",
                                "trainer": "NODE",
                                "n_epochs": 5,
                            },
                        ],
                    }
                }
            ],
        }
    )

    assert isinstance(specs[0], LinearSolvePhaseSpec)
    assert isinstance(specs[1], OptimizerPhaseSpec)
    assert isinstance(specs[2], LinearSolvePhaseSpec)
    assert isinstance(specs[3], OptimizerPhaseSpec)
    assert isinstance(specs[-4], AnalysisPhaseSpec)
    assert isinstance(specs[-3], ExportPhaseSpec)
    assert isinstance(specs[-2], ExportPhaseSpec)
    assert isinstance(specs[-1], ExportPhaseSpec)


def test_normalize_phase_specs_warns_for_analysis_and_export_inside_repeat():
    with pytest.warns(UserWarning, match="Repeat block 'cycle' contains an analysis phase"):
        with pytest.warns(UserWarning, match="Repeat block 'cycle' contains an export phase"):
            specs = normalize_phase_specs(
                {
                    "model": {"name": "demo"},
                    "phases": [
                        {
                            "repeat": {
                                "name": "cycle",
                                "times": 1,
                                "phases": [
                                    {"type": "analysis", "name": "inspect"},
                                    {
                                        "type": "export",
                                        "name": "save_model",
                                        "export_kind": "best_model",
                                    },
                                ],
                            }
                        }
                    ],
                }
            )

    assert isinstance(specs[0], AnalysisPhaseSpec)
    assert isinstance(specs[1], ExportPhaseSpec)


def test_normalize_phase_specs_accepts_smoothing_data_phase():
    specs = normalize_phase_specs(
        {
            "model": {"name": "demo"},
            "phases": [
                {
                    "type": "data",
                    "name": "smooth",
                    "operation": "smooth",
                    "method": "savgol",
                    "window_length": 5,
                    "polyorder": 2,
                }
            ],
        }
    )

    data_spec = specs[0]
    assert isinstance(data_spec, DataPhaseSpec)
    assert data_spec.operation == "smooth"
    assert data_spec.config["method"] == "savgol"


def test_normalize_phase_specs_rejects_legacy_ls_update():
    with pytest.raises(
        PhaseSpecValidationError, match="'ls_update' is deprecated and no longer supported"
    ):
        normalize_phase_specs(
            {
                "model": {"name": "demo"},
                "phases": [
                    {
                        "name": "node",
                        "trainer": "NODE",
                        "n_epochs": 5,
                        "ls_update": {"method": "truncated", "params": 2},
                    }
                ],
            }
        )


def test_validation_analysis_phase_uses_phase_solver_settings(monkeypatch):
    config = {
        "model": {"name": "demo"},
        "phases": [
            {
                "type": "optimizer",
                "name": "node",
                "trainer": "NODE",
                "ode_method": "rk4",
                "ode_args": {"step_size": 0.05},
            }
        ],
    }
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        AnalysisPhaseSpec(name="analysis"),
        config=config,
        model_class=object,
        dtype=torch.float32,
        execution_services=execution_services,
    )

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    model = _DummyModel()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criteria=[torch.nn.MSELoss(), torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
    )
    artifacts = ArtifactRegistry()
    artifacts.put(
        "model",
        ModelArtifact(
            model=model,
            config=config,
            train_md={},
            valid_md={},
            dtype=torch.float32,
        ),
    )
    artifacts.put("optimizer_state", optimizer_state)
    trainer_state = TrainerState(config=config, device=torch.device("cpu"))
    phase_context = PhaseContext(train_set=[object()], valid_set=[object()])

    captured: dict[str, object] = {}

    def traced_eval(
        self,
        model,
        optimizer_state,
        dataset,
        *,
        method,
        ode_args=None,
        evaluate_all=False,
    ):
        captured["method"] = method
        captured["ode_args"] = ode_args
        captured["dataset"] = dataset
        captured["evaluate_all"] = evaluate_all
        return 0.25

    monkeypatch.setattr(
        phases_module.NodeOptimizerPhase,
        "_evaluate_prediction_criterion",
        traced_eval,
    )

    result = phase.execute(
        trainer_state=trainer_state,
        phase_context=phase_context,
        artifacts=artifacts,
        run_name="demo",
        logger=execution_services.configure_logger("dymad.test.validation_analysis", prefix=""),
    )

    assert captured["method"] == "rk4"
    assert captured["ode_args"] == {"step_size": 0.05}
    assert captured["dataset"] is phase_context.valid_set
    assert captured["evaluate_all"] is False
    assert result.metrics == {"mse": 0.25}


def test_smoothing_data_phase_rewrites_regular_context_and_loader():
    train_series = _build_regular_series()
    valid_series = _build_regular_series(offset=0.2)
    original_context = PhaseContext(
        train_set=[train_series],
        valid_set=[valid_series],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth",
            operation="smooth",
            config={"method": "savgol", "window_length": 5, "polyorder": 2},
        ),
    )

    result = phase.execute(
        trainer_state=TrainerState(config=config, device=torch.device("cpu")),
        phase_context=original_context,
        artifacts=ArtifactRegistry(),
        run_name="demo",
        logger=logging.getLogger("test.smooth.regular"),
    )

    assert result.phase_context.train_set is not original_context.train_set
    assert result.phase_context.valid_set is not original_context.valid_set
    assert isinstance(next(iter(result.phase_context.train_loader)), RegularTrainerBatch)
    np.testing.assert_allclose(
        result.phase_context.train_set[0].state.detach().cpu().numpy(),
        savgol_filter(train_series.state.detach().cpu().numpy(), 5, 2, axis=0),
    )
    train_delta = (
        result.phase_context.train_set[0].state.detach().cpu().numpy()
        - train_series.state.detach().cpu().numpy()
    )
    assert result.metrics["train_delta_rmse"] == pytest.approx(
        float(np.sqrt(np.mean(np.square(train_delta))))
    )
    assert "train_num_trajectories" not in result.metrics
    assert "train_num_elements" not in result.metrics
    assert "train_num_diff_elements" not in result.metrics
    assert result.metrics["train_delta_mae"] == pytest.approx(float(np.mean(np.abs(train_delta))))
    assert result.metrics["train_roughness_ratio"] == pytest.approx(
        result.metrics["train_roughness_after"] / result.metrics["train_roughness_before"]
    )
    assert result.record is not None
    assert result.record.metrics["train_delta_rmse"] == pytest.approx(
        result.metrics["train_delta_rmse"]
    )
    assert result.phase_context.train_md["data_phase_history"][-1]["method"] == "savgol"
    assert result.phase_context.train_md["data_phase_history"][-1]["metrics"][
        "train_delta_rmse"
    ] == pytest.approx(result.metrics["train_delta_rmse"])
    assert result.phase_context.valid_md["data_phase_history"][-1]["split"] == "valid"


def test_smoothing_data_phase_logs_train_valid_table(caplog):
    train_series = _build_regular_series()
    valid_series = _build_regular_series(offset=0.2)
    context = PhaseContext(
        train_set=[train_series],
        valid_set=[valid_series],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth",
            operation="smooth",
            config={"method": "savgol", "window_length": 5, "polyorder": 2},
        ),
    )

    with caplog.at_level(logging.INFO):
        phase.execute(
            trainer_state=TrainerState(config=config, device=torch.device("cpu")),
            phase_context=context,
            artifacts=ArtifactRegistry(),
            run_name="demo",
            logger=logging.getLogger("test.smooth.table"),
        )

    assert "Data phase 'smooth' completed:" in caplog.text
    assert "metric" in caplog.text
    assert "train" in caplog.text
    assert "valid" in caplog.text
    assert "delta_rmse" in caplog.text
    assert "roughness_ratio" in caplog.text
    assert "num_trajectories" not in caplog.text
    assert "num_elements" not in caplog.text


def test_smoothing_data_phase_supports_train_only_split():
    train_series = _build_regular_series()
    valid_series = _build_regular_series(offset=0.2)
    original_context = PhaseContext(
        train_set=[train_series],
        valid_set=[valid_series],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth_train",
            operation="smooth",
            config={"splits": ["train"], "window_length": 5, "polyorder": 2},
        ),
    )

    result = phase.execute(
        trainer_state=TrainerState(config=config, device=torch.device("cpu")),
        phase_context=original_context,
        artifacts=ArtifactRegistry(),
        run_name="demo",
        logger=logging.getLogger("test.smooth.train"),
    )

    assert result.phase_context.valid_set is original_context.valid_set
    assert result.phase_context.valid_loader is original_context.valid_loader
    assert "data_phase_history" not in result.phase_context.valid_md
    assert result.phase_context.train_set is not original_context.train_set
    assert "valid_delta_rmse" not in result.metrics
    assert "train_delta_rmse" in result.metrics


def test_smoothing_data_phase_supports_graph_node_state_only():
    graph_series = _build_graph_series()
    original_context = PhaseContext(
        train_set=[graph_series],
        valid_set=[graph_series],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth_graph",
            operation="smooth",
            config={"splits": ["valid"], "window_length": 5, "polyorder": 2},
        ),
    )

    result = phase.execute(
        trainer_state=TrainerState(config=config, device=torch.device("cpu")),
        phase_context=original_context,
        artifacts=ArtifactRegistry(),
        run_name="demo",
        logger=logging.getLogger("test.smooth.graph"),
    )

    smoothed = result.phase_context.valid_set[0]
    np.testing.assert_allclose(
        smoothed.node_state.detach().cpu().numpy(),
        savgol_filter(graph_series.node_state.detach().cpu().numpy(), 5, 2, axis=0),
    )
    assert torch.equal(smoothed.edge_index, graph_series.edge_index)
    assert torch.equal(smoothed.edge_weight, graph_series.edge_weight)
    assert result.phase_context.train_set is original_context.train_set


@pytest.mark.parametrize(
    "phase_config,match",
    [
        ({"window_length": 4, "polyorder": 2}, "odd, positive window_length"),
        ({"window_length": 5, "polyorder": 5}, "polyorder < window_length"),
        ({"window_length": 11, "polyorder": 2}, "at least 11 steps"),
        (
            {"method": "median", "window_length": 5, "polyorder": 2},
            "Unsupported data smoothing method",
        ),
        ({"splits": ["test"], "window_length": 5, "polyorder": 2}, "invalid splits"),
    ],
)
def test_smoothing_data_phase_rejects_invalid_configs(phase_config, match):
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth_invalid",
            operation="smooth",
            config=phase_config,
        ),
    )
    context = PhaseContext(
        train_set=[_build_regular_series()],
        valid_set=[_build_regular_series(offset=0.2)],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )

    with pytest.raises(PhaseSpecValidationError, match=match):
        phase.execute(
            trainer_state=TrainerState(config=config, device=torch.device("cpu")),
            phase_context=context,
            artifacts=ArtifactRegistry(),
            run_name="demo",
            logger=logging.getLogger("test.smooth.invalid"),
        )


def test_load_config_normalizes_legacy_training_alias(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
training:
  ode_method: rk4
  ode_args:
    step_size: 0.2
phases:
  - type: optimizer
    name: node
    trainer: NODE
    ode_method: dopri5
    ode_args:
      rtol: 1.0e-6
cv:
  param_grid:
    training.learning_rate: [0.1, 0.2]
  search:
    mode: nelder_mead_like
    bounds:
      training.weight_decay: [1.0e-4, 1.0e-2]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert "training" not in config
    assert config["phases"][0]["ode_method"] == "rk4"
    assert config["phases"][0]["ode_args"] == {"step_size": 0.2}
    assert config["cv"]["param_grid"] == {"phases.0.learning_rate": [0.1, 0.2]}
    assert config["cv"]["search"]["bounds"] == {"phases.0.weight_decay": [1.0e-4, 1.0e-2]}

    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        AnalysisPhaseSpec(name="analysis"),
        config=config,
        model_class=object,
        dtype=torch.float32,
        execution_services=execution_services,
    )

    assert phase._prediction_settings() == ("rk4", {"step_size": 0.2})


def test_run_cv_single_uses_trainer_run_with_typed_context(monkeypatch):
    calls = {"init": 0, "run": 0}
    expected_metric = 0.123
    phase_context = object()
    trainer_state = object()

    cfg = {
        "model": {"name": "demo-model"},
        "path": {"checkpoint_prefix": "/tmp/cp", "results_prefix": "/tmp/rp"},
        "phases": [{"name": "p0", "trainer": "Linear"}],
    }

    monkeypatch.setattr(
        driver,
        "_apply_combo_to_config",
        lambda combo_idx, fold_id, fold_cfg, combo, base_name, checkpoint_prefix, results_prefix: (
            cfg,
            "/tmp/model_prefix",
        ),
    )
    monkeypatch.setattr(
        driver, "_build_phase_context", lambda fold_id, cfg, train_sets, valid_sets: phase_context
    )
    monkeypatch.setattr(
        driver, "build_initial_trainer_state", lambda cfg, execution_services: trainer_state
    )

    class _FakePhaseResult:
        def get_metric(self, metric_name):
            assert metric_name == "total"
            return expected_metric

    class _FakeTrainerRun:
        def __init__(
            self,
            config,
            model_class,
            device,
            dtype,
            run_name,
            checkpoint_prefix,
            results_prefix,
            execution_services=None,
        ):
            calls["init"] += 1
            calls["run_name"] = run_name
            calls["checkpoint_prefix"] = checkpoint_prefix
            calls["results_prefix"] = results_prefix
            calls["execution_services"] = execution_services

        def run(self, *, initial_context, initial_state, artifacts=None):
            calls["run"] += 1
            calls["initial_context"] = initial_context
            calls["initial_state"] = initial_state
            return [_FakePhaseResult()]

    monkeypatch.setattr(driver, "TrainerRun", _FakeTrainerRun)

    class _FakeTrainSet:
        dtype = torch.float32

    args = {
        "combo_idx": 5,
        "fold_idx": 2,
        "fold_cfg": {"seed": 0},
        "combo": {"phases.0.lr": 0.1},
        "base_name": "base",
        "checkpoint_prefix": "/checkpoints",
        "results_prefix": "/results",
        "train_sets": [_FakeTrainSet()],
        "valid_sets": [_FakeTrainSet()],
        "model_class": object,
        "device": torch.device("cpu"),
        "metric": "total",
    }

    result = driver.run_cv_single(args)

    assert calls["init"] == 1
    assert calls["run"] == 1
    assert calls["initial_context"] is phase_context
    assert calls["initial_state"] is trainer_state
    assert calls["run_name"] == "demo-model"
    assert calls["checkpoint_prefix"] == "/tmp/cp"
    assert calls["results_prefix"] == "/tmp/rp"
    assert calls["execution_services"] is not None
    assert result["metric_value"] == expected_metric


def test_select_best_cv_result_uses_goal_and_tie_breakers() -> None:
    cv_results = [
        CVResult(params={"model.koopman_dimension": 6}, fold_metrics=[1.0, 1.0]),
        CVResult(params={"model.koopman_dimension": 4}, fold_metrics=[1.0, 1.2]),
        CVResult(params={"model.koopman_dimension": 8}, fold_metrics=[1.1, 1.1]),
    ]

    best_idx = select_best_cv_result(
        cv_results,
        goal="minimize",
        tie_breakers=("std_metric", "param_l1", "combo_index"),
    )

    assert best_idx == 0


def test_select_best_cv_result_supports_maximize_goal() -> None:
    cv_results = [
        CVResult(params={"model.koopman_dimension": 6}, fold_metrics=[1.0, 1.0]),
        CVResult(params={"model.koopman_dimension": 4}, fold_metrics=[2.0, 2.0]),
    ]

    best_idx = select_best_cv_result(
        cv_results,
        goal="maximize",
        tie_breakers=("std_metric", "combo_index"),
    )

    assert best_idx == 1


def test_select_best_cv_result_uses_explicit_combo_indices_for_tie_breaker() -> None:
    cv_results = [
        CVResult(params={"model.koopman_dimension": 8}, fold_metrics=[1.0]),
        CVResult(params={"model.koopman_dimension": 5}, fold_metrics=[1.0]),
        CVResult(params={"model.koopman_dimension": 3}, fold_metrics=[1.0]),
    ]

    best_idx = select_best_cv_result(
        cv_results,
        goal="minimize",
        tie_breakers=("std_metric", "combo_index"),
        combo_indices=(8, 5, 3),
    )

    assert best_idx == 2


def test_select_best_cv_result_rejects_unsupported_tie_breaker() -> None:
    cv_results = [CVResult(params={"model.koopman_dimension": 4}, fold_metrics=[1.0])]

    with pytest.raises(ValueError, match="unsupported tie breaker"):
        select_best_cv_result(cv_results, tie_breakers=("unknown_rule",))


def test_nelder_mead_like_search_indices_limits_evaluations() -> None:
    combos = [{"model.koopman_dimension": value} for value in range(9)]
    evaluation_calls: list[int] = []

    def _evaluate(index: int) -> float:
        evaluation_calls.append(index)
        value = float(combos[index]["model.koopman_dimension"])
        return (value - 5.0) ** 2

    evaluated = nelder_mead_like_search_indices(
        combos,
        evaluate_index=_evaluate,
        max_iterations=1,
    )

    assert evaluated == evaluation_calls
    assert evaluated[:2] == [0, 8]
    assert len(evaluated) == 4
    assert 6 in evaluated


def test_nelder_mead_like_search_indices_falls_back_to_grid_for_non_numeric_values() -> None:
    combos = [{"model.variant": name} for name in ("a", "b", "c")]
    evaluation_calls: list[int] = []

    def _evaluate(index: int) -> float:
        evaluation_calls.append(index)
        return float(index)

    evaluated = nelder_mead_like_search_indices(
        combos,
        evaluate_index=_evaluate,
        max_iterations=3,
    )

    assert evaluated == [0, 1, 2]
    assert evaluation_calls == [0, 1, 2]


def test_bounded_nelder_mead_search_points_respects_bounds() -> None:
    target = np.array([0.25, 0.75], dtype=float)

    def _evaluate(point: np.ndarray) -> float:
        return float(np.sum((point - target) ** 2))

    evaluated = bounded_nelder_mead_search_points(
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
        evaluate_point=_evaluate,
        max_iterations=6,
    )

    assert evaluated
    for point in evaluated:
        assert np.all(point >= 0.0)
        assert np.all(point <= 1.0)
    best_point = min(evaluated, key=_evaluate)
    assert np.linalg.norm(best_point - target) <= 0.25


def test_single_split_driver_train_supports_nelder_mead_like_search(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
phases:
  - trainer: Linear
cv:
  param_grid:
    model.koopman_dimension: [0, 1, 2, 3, 4]
  search:
    mode: nelder_mead_like
    max_iterations: 1
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    evaluated_dims: list[int] = []

    def _fake_run_cv_single(args):
        value = int(args["combo"]["model.koopman_dimension"])
        evaluated_dims.append(value)
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(f"{model_prefix}_summary.npz", koopman_dimension=np.array([value]))
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": float((value - 2) ** 2),
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    trainer = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )
    _, best_result, all_results = trainer.train()

    assert len(all_results) == 4
    assert len(evaluated_dims) == 4
    assert best_result.params["model.koopman_dimension"] == 2


def test_single_split_driver_train_supports_explicit_grid_search(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
phases:
  - trainer: Linear
cv:
  param_grid:
    model.koopman_dimension: [0, 1, 2, 3, 4]
  search:
    mode: grid
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    evaluated_dims: list[int] = []

    def _fake_run_cv_single(args):
        value = int(args["combo"]["model.koopman_dimension"])
        evaluated_dims.append(value)
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(f"{model_prefix}_summary.npz", koopman_dimension=np.array([value]))
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": float((value - 2) ** 2),
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    trainer = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )
    _, best_result, all_results = trainer.train()

    assert evaluated_dims == [0, 1, 2, 3, 4]
    assert len(all_results) == 5
    assert best_result.params["model.koopman_dimension"] == 2


def test_single_split_driver_train_supports_bounded_nelder_mead_search(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
  koopman_dimension: 1
training:
  weak_form_params:
    N: 13
phases:
  - trainer: Weak
cv:
  search:
    mode: nelder_mead_like
    bounds:
      model.koopman_dimension: [0, 4]
      training.weak_form_params.N:
        lower: 9
        upper: 17
        parity: odd
    max_iterations: 6
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    evaluated_params: list[tuple[int, int]] = []

    def _fake_run_cv_single(args):
        koopman_dimension = int(args["combo"]["model.koopman_dimension"])
        weak_n = int(args["combo"]["phases.0.weak_form_params.N"])
        evaluated_params.append((koopman_dimension, weak_n))
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(
            f"{model_prefix}_summary.npz",
            koopman_dimension=np.array([koopman_dimension]),
            weak_n=np.array([weak_n]),
        )
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": float((koopman_dimension - 2) ** 2 + 0.01 * (weak_n - 11) ** 2),
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    trainer = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )
    _, best_result, all_results = trainer.train()

    assert all_results
    assert evaluated_params
    assert all(0 <= koopman_dimension <= 4 for koopman_dimension, _ in evaluated_params)
    assert all(9 <= weak_n <= 17 and weak_n % 2 == 1 for _, weak_n in evaluated_params)
    assert best_result.params["model.koopman_dimension"] == 2
    assert best_result.params["phases.0.weak_form_params.N"] == 11


def test_single_split_driver_bounded_nelder_mead_evaluates_integer_upper_endpoint(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
  koopman_dimension: 1
phases:
  - trainer: Linear
cv:
  search:
    mode: nelder_mead_like
    bounds:
      model.koopman_dimension: [0, 3]
    max_iterations: 4
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    evaluated_dims: list[int] = []

    def _fake_run_cv_single(args):
        value = int(args["combo"]["model.koopman_dimension"])
        evaluated_dims.append(value)
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(f"{model_prefix}_summary.npz", koopman_dimension=np.array([value]))
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": float((value - 3) ** 2),
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    trainer = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )
    _, best_result, _ = trainer.train()

    assert 3 in evaluated_dims
    assert best_result.params["model.koopman_dimension"] == 3


@pytest.mark.parametrize(
    ("config_text", "expected_error", "expected_match"),
    [
        (
            """
model:
  name: demo
phases:
  - trainer: Linear
    learning_rate: 0.1
cv:
  search:
    mode: nelder_mead_like
    bounds:
      phases.0.learning_rate:
        lower: 0.01
        upper: 0.2
        parity: odd
""".strip(),
            TypeError,
            "parity is only supported for integer-valued config fields",
        ),
        (
            """
model:
  name: demo
phases:
  - trainer: Linear
cv:
  search:
    mode: nelder_mead_like
    bounds:
      model.name: [0, 4]
""".strip(),
            TypeError,
            "must target an integer or floating-point config value",
        ),
        (
            """
model:
  name: demo
phases:
  - trainer: Linear
cv:
  search:
    mode: nelder_mead_like
    bounds:
      model.missing_dimension: [0, 4]
""".strip(),
            TypeError,
            "does not resolve in the config",
        ),
    ],
)
def test_single_split_driver_rejects_invalid_bounded_nelder_mead_yaml_configs(
    monkeypatch, tmp_path, config_text, expected_error, expected_match
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    with pytest.raises(expected_error, match=expected_match):
        driver.SingleSplitDriver(
            config_path=str(config_path),
            model_class=torch.nn.Module,
            device=torch.device("cpu"),
        )


def test_single_split_driver_train_uses_grid_combo_index_for_tie_breaker(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
phases:
  - trainer: Linear
cv:
  param_grid:
    model.koopman_dimension: [0, 1, 2, 3, 4, 5, 6, 7, 8]
  search:
    mode: nelder_mead_like
    max_iterations: 2
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)
    def _fake_nelder_mead_like_search_indices(combos, *, evaluate_index, **kwargs):
        evaluated = [0, 8, 2, 5, 7, 4, 1, 3]
        for index in evaluated:
            evaluate_index(index)
        return evaluated

    monkeypatch.setattr(
        driver,
        "nelder_mead_like_search_indices",
        _fake_nelder_mead_like_search_indices,
    )

    evaluated_dims: list[int] = []

    def _fake_run_cv_single(args):
        value = int(args["combo"]["model.koopman_dimension"])
        evaluated_dims.append(value)
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(f"{model_prefix}_summary.npz", koopman_dimension=np.array([value]))
        metric_value = 0.0 if value in {8, 5, 3} else 1.0
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": metric_value,
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    trainer = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )
    _, best_result, all_results = trainer.train()

    assert evaluated_dims == [0, 8, 2, 5, 7, 4, 1, 3]
    assert len(all_results) == 8
    assert best_result.params["model.koopman_dimension"] == 3


def test_trainer_run_rejects_legacy_resume_checkpoints(tmp_path):
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"config": {"model": {"name": "demo"}}}, checkpoint_path)

    trainer_run = TrainerRun(
        config={"model": {"name": "demo"}, "phases": [{"name": "p0", "trainer": "Linear"}]},
        model_class=object,
        device=torch.device("cpu"),
        dtype=torch.float32,
        run_name="demo",
        checkpoint_prefix=str(tmp_path),
        results_prefix=str(tmp_path),
    )

    try:
        trainer_run.load_run_checkpoint(str(checkpoint_path))
    except TrainingCheckpointError as exc:
        assert "Legacy optimizer checkpoints are not resumable" in str(exc)
    else:
        raise AssertionError("expected TrainingCheckpointError for legacy checkpoint payload")


def test_trainer_run_round_trips_typed_run_checkpoint(tmp_path):
    trainer_run = TrainerRun(
        config={"model": {"name": "demo"}, "phases": [{"name": "p0", "trainer": "Linear"}]},
        model_class=object,
        device=torch.device("cpu"),
        dtype=torch.float32,
        run_name="demo",
        checkpoint_prefix=str(tmp_path),
        results_prefix=str(tmp_path),
    )
    state = TrainerState(
        config={"model": {"name": "demo"}},
        device=torch.device("cpu"),
        epoch=3,
        best_loss={"valid_total": 1.23},
        phase_cursor=2,
        phase_records=[
            PhaseRecord(name="p0", kind="optimizer", started_epoch=0, completed_epoch=3)
        ],
    )
    artifacts = ArtifactRegistry()
    trainer_run.save_run_checkpoint(state, artifacts)

    loaded_state, loaded_artifacts = trainer_run.load_run_checkpoint()

    assert loaded_state.epoch == 3
    assert loaded_state.phase_cursor == 2
    assert list(loaded_artifacts.keys()) == []


def test_trainer_run_replays_completed_data_phases_on_resume(monkeypatch, tmp_path):
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [
            {
                "type": "data",
                "name": "smooth",
                "operation": "smooth",
                "window_length": 5,
                "polyorder": 2,
            },
            {"name": "linear", "trainer": "Linear"},
        ],
    }
    trainer_run = TrainerRun(
        config=config,
        model_class=object,
        device=torch.device("cpu"),
        dtype=torch.float32,
        run_name="demo",
        checkpoint_prefix=str(tmp_path),
        results_prefix=str(tmp_path),
    )
    initial_context = PhaseContext(
        train_set=[_build_regular_series()],
        valid_set=[_build_regular_series(offset=0.2)],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )
    resumed_state = TrainerState(
        config=config,
        device=torch.device("cpu"),
        phase_cursor=1,
    )
    monkeypatch.setattr(trainer_run, "_maybe_resume", lambda: (resumed_state, ArtifactRegistry()))

    captured = {}

    def fake_run(*, initial_context, initial_state, artifacts, run_name, checkpoint_callback):
        captured["initial_context"] = initial_context
        captured["initial_state"] = initial_state
        return []

    monkeypatch.setattr(trainer_run.pipeline, "run", fake_run)

    trainer_run.run(initial_context=initial_context)

    replayed_context = captured["initial_context"]
    np.testing.assert_allclose(
        replayed_context.train_set[0].state.detach().cpu().numpy(),
        savgol_filter(initial_context.train_set[0].state.detach().cpu().numpy(), 5, 2, axis=0),
    )
    assert replayed_context.train_md["data_phase_history"][-1]["phase"] == "smooth"
    assert captured["initial_state"] is resumed_state


def test_summary_export_persists_phase_metrics(tmp_path):
    config = {
        "model": {"name": "demo"},
        "plotting": {"prediction": False},
        "path": {"checkpoint_prefix": str(tmp_path), "results_prefix": str(tmp_path)},
        "phases": [],
    }
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        ExportPhaseSpec(name="export_summary", export_kind="summary"),
        config=config,
        model_class=torch.nn.Linear,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    trainer_state = TrainerState(
        config=config,
        device=torch.device("cpu"),
        best_loss={"valid_total": 0.3},
        phase_records=[
            PhaseRecord(
                name="smooth",
                kind="data",
                started_epoch=0,
                completed_epoch=0,
                metrics={"train_delta_rmse": 0.1, "train_roughness_ratio": 0.8},
            )
        ],
    )
    artifacts = ArtifactRegistry()
    artifacts.put(
        "model",
        ModelArtifact(
            model=torch.nn.Linear(2, 2),
            config=config,
            train_md={},
            valid_md={},
            dtype=torch.float32,
        ),
    )
    artifacts.put(
        "history",
        TrainingHistoryArtifact(
            hist=[{"train_total": [0.2], "valid_total": [0.3]}],
            epoch_times=[1.0],
            best_loss={"valid_total": 0.3},
        ),
    )

    result = phase.execute(
        trainer_state=trainer_state,
        phase_context=PhaseContext(),
        artifacts=artifacts,
        run_name="demo",
        logger=logging.getLogger("test.export.summary"),
    )

    assert result.metrics["exports_written"] >= 1.0
    with np.load(tmp_path / "demo_summary.npz", allow_pickle=True) as npz:
        phase_metrics = npz["phase_metrics"].item()
        phase_records = npz["phase_records"].tolist()

    assert phase_metrics["smooth"]["train_delta_rmse"] == pytest.approx(0.1)
    assert phase_metrics["smooth"]["train_roughness_ratio"] == pytest.approx(0.8)
    assert len(phase_records) == 1
    assert phase_records[0].name == "smooth"
